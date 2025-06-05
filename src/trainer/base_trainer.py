"""Core functionality for self-refining trainer (model init, sampling, etc.)"""

import inspect
import os.path as osp
from functools import partial
from random import shuffle

import jax
import jax.numpy as jnp
import optax
from loguru import logger as log
from omegaconf import OmegaConf
from orbax.checkpoint.checkpoint_utils import construct_restore_args
from tqdm import tqdm

from src.commons.graph import batch_data, build_graph
from src.models.components.energy import predict_energy
from src.trainer.buffer import Buffer
from src.trainer.utils import TrainState, count_parameters, init_checkpoint_manager


def ensure_list(samples):
    return samples if isinstance(samples, list) else [samples]


def loss_fn(params, apply_fn, graph, output_coefficients=True):
    energy_output, C = predict_energy(
        params, apply_fn, graph, output_coefficients=output_coefficients
    )
    return energy_output.mean(), C


@jax.jit
def train_step(state, graph):
    (loss, C), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.apply_fn, graph, output_coefficients=True
    )
    state = state.apply_gradients(grads=grads)
    grad_norm = optax.global_norm(grads)
    return state, loss, grad_norm


@jax.jit
def eval_step(state, graph):
    energy_output = jax.lax.stop_gradient(
        predict_energy(state.params, state.apply_fn, graph, output_coefficients=False)
    )
    return energy_output.mean()


class BaseTrainer:
    """
    Common base class for SupervisedTrainer and SelfRefiningTrainer.
    """

    def __init__(
        self,
        key,
        model,
        datamodule,
        optimizer,
        scheduler,
        ckpt_dir=None,
        logger=None,
        # common kwargs
        basis_name="sto-3g",
        xc_method="lda",
        grid_level=3,
        num_iterations=100_000,
        num_recycle_steps: int = 10,
        num_data_samples: int = 25000,
        batch_size: int = 8,
        eval_freq: int = 5000,
        save_every: int = 1000,
        **kwargs,
    ):
        self.save_hyperparameters(
            ignore=("model", "optimizer", "datamodule", "scheduler", "logger", "key")
        )

        self.key = key
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        self.train_data_iter = self.init_dataloader(
            size=num_data_samples, batch_size=batch_size
        )

        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            self.checkpoint_manager = init_checkpoint_manager(ckpt_dir, config={})
            self.latest_checkpoint_manager = init_checkpoint_manager(
                ckpt_dir, config={}, keep_latest=True
            )
            log.info(f"Saving model checkpoints at {self.checkpoint_manager.directory}")
        else:
            self.checkpoint_manager = None
            self.latest_checkpoint_manager = None

    def save_hyperparameters(self, ignore=()):
        """Capture all __init__ args (except those in ignore) into self.hparams."""
        frame = inspect.currentframe().f_back
        local_vars = {
            k: v
            for k, v in frame.f_locals.items()
            if k != "self" and k != "__class__" and k not in ignore
        }

        self.hparams = OmegaConf.create(local_vars)

    def init_dataloader(self, size, batch_size):
        return iter(self.datamodule.train_dataloader(size=size, batch_size=batch_size))

    def initialize_params(self):
        sample = self.datamodule.valid_dataloader().dataset[0]
        self._atomic_number = sample.atomic_number

        params = self.model.init(
            self.key,
            atomic_number=sample.atomic_number,
            position=sample.position,
            orbital_tokens=sample.orbital_tokens,
            orbital_index=sample.orbital_index,
            hamiltonian=sample.hamiltonian,
            senders=sample.senders,
            receivers=sample.receivers,
        )
        num_params = count_parameters(params)
        log.info(f"Parameter count: {num_params}")
        if self.logger:
            self.logger.log_hyperparameters({"model/params/total": num_params})
        return params

    def initialize_state(self):
        params = self.initialize_params()
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            key=self.key,
            step_size=getattr(self, "step_size", 1e-3),
            num_generated=getattr(self, "num_generated", 0),
        )
        return state

    def initialize_model(self):
        state = self.initialize_state()
        state = self.restore_from_checkpoint(state)
        return state

    def restore_from_checkpoint(self, state, use_latest=True):
        checkpoint_manager = (
            self.latest_checkpoint_manager if use_latest else self.checkpoint_manager
        )
        if checkpoint_manager is not None:
            step = (
                checkpoint_manager.latest_step()
                if use_latest
                else checkpoint_manager.best_step()
            )
            if step is not None:
                restore_args = construct_restore_args(state)
                state = checkpoint_manager.restore(
                    step,
                    items=state,
                    restore_kwargs={"restore_args": restore_args},
                )
                self.key = state.key
                state = state.replace(step=step)
                log.info(f"Restored from checkpoint at step {state.step}")
                # ensure next call starts one step after:
                state = state.replace(step=state.step + 1)

        return state

    def validate(self, state):
        valid_losses = []
        valid_dataloader = self.datamodule.valid_dataloader()

        for graph in (pbar := tqdm(valid_dataloader, leave=False)):
            graph = batch_data(graph)
            valid_loss_step = eval_step(state, graph)
            valid_losses.append(valid_loss_step)
            pbar.set_description(f"Validation Loss: {valid_loss_step:06f}")

        return jnp.mean(jnp.array(valid_losses)).item()


class BaseSelfRefiningTrainer(BaseTrainer):
    def __init__(
        self,
        key,
        model,
        datamodule,
        optimizer,
        scheduler,
        sampler,
        basis_name="sto-3g",
        xc_method="lda",
        grid_level=3,
        num_iterations=100_000,
        num_recycle_steps: int = 10,
        batch_size=32,
        training_buffer_max_size=256,
        num_data_samples=None,
        sample_buffer_prob=0.9,
        num_pretrain_steps: int = 0,
        num_init_samples: int = 128,
        eval_freq: int = 5000,
        save_every: int = 1000,
        ckpt_dir=None,
        logger=None,
    ):
        super().__init__(
            key,
            model,
            datamodule,
            optimizer,
            scheduler,
            basis_name=basis_name,
            xc_method=xc_method,
            grid_level=grid_level,
            num_iterations=num_iterations,
            num_recycle_steps=num_recycle_steps,
            batch_size=batch_size,
            eval_freq=eval_freq,
            save_every=save_every,
            ckpt_dir=ckpt_dir,
            logger=logger,
        )

        self.save_hyperparameters(
            ignore=(
                "model",
                "sampler",
                "optimizer",
                "datamodule",
                "scheduler",
                "logger",
                "key",
            )
        )

        self.train_device, self.sampler_device = jax.devices()[:2]
        atomic_number = datamodule.train_dataloader().dataset[0].atomic_number

        # sampler on its own device:
        self.sampler = sampler(
            model_apply_fn=self.model.apply,
            init_params=0,  # dummy params
            sample_fn=partial(self.sample_mixed_batch, use_sampler_iter=True),
            atomic_number=atomic_number,
            device=self.sampler_device,
        )

        self.num_generated = 0
        self.training_buffer = Buffer(max_size=training_buffer_max_size)

        # two iterators: one for train and one for sampler
        self.train_data_iter = self.init_dataloader(
            size=num_data_samples, batch_size=self.hparams.batch_size
        )
        self.sampler_data_iter = self.init_dataloader(
            size=num_data_samples, batch_size=self.sampler.batch_size
        )

        # path for saving the buffer (so we can reload it):
        self.buffer_save_path = (
            osp.join(ckpt_dir, "replay_buffer.npy") if ckpt_dir else "buffer.npy"
        )

    def initialize_model(self):
        with jax.default_device(self.train_device):
            return super().initialize_model()

    def restore_from_checkpoint(self, state, use_latest=True):
        state = super().restore_from_checkpoint(state, use_latest=use_latest)
        checkpoint_manager = self.latest_checkpoint_manager if use_latest else self.checkpoint_manager
        if checkpoint_manager is not None:
            self.key = jax.device_put(state.key, device=self.train_device)

            self.step_size = getattr(state, "step_size", 1e-3)
            self.num_generated = getattr(state, "num_generated", 0)

            # If we have a saved buffer file, load it:
            if osp.exists(self.buffer_save_path):
                self.training_buffer.load(self.buffer_save_path)

            # If buffer has more samples than state.num_generated, update that:
            self.num_generated = max(self.num_generated, len(self.training_buffer))

            log.info(f"Total generated so far: {self.num_generated}")

        return state

    def sample_from_training_buffer(self, batch_size=32):
        pos_list = self.training_buffer.sample(batch_size)
        return [
            build_graph(
                atomic_number=self._atomic_number,
                position=pos,
                basis_name=self.hparams.basis_name,
                xc_method=self.hparams.xc_method,
                grid_level=self.hparams.grid_level,
            )
            for pos in pos_list
        ]

    def sample_mixed_batch(self, batch_size=32, use_sampler_iter=False):
        data_iter = self.sampler_data_iter if use_sampler_iter else self.train_data_iter
        total_samples = self.hparams.num_data_samples + self.num_generated
        p_buffer = self.num_generated / max(total_samples, 1)

        # Cap the data side so that p(buffer) ≤ sample_buffer_prob
        if p_buffer > self.hparams.sample_buffer_prob:
            total_samples = int(self.hparams.sample_buffer_prob * total_samples)

        self.key, key = jax.random.split(self.key)
        idx = jax.random.choice(key, total_samples, shape=(batch_size,), replace=False)

        num_data = int(jnp.sum(idx < self.hparams.num_data_samples))
        num_buffer = batch_size - num_data

        mixed = []
        if num_data > 0:
            samples = next(data_iter, None)
            if samples is None:
                # re-initialize the appropriate iterator
                if use_sampler_iter:
                    self.sampler_data_iter = self.init_dataloader(
                        size=self.hparams.num_data_samples,
                        batch_size=self.sampler.batch_size,
                    )
                    samples = next(self.sampler_data_iter)
                else:
                    self.train_data_iter = self.init_dataloader(
                        size=self.hparams.num_data_samples,
                        batch_size=self.hparams.batch_size,
                    )
                    samples = next(self.train_data_iter)

            samples = ensure_list(samples)
            mixed.extend(samples[:num_data])

        if num_buffer > 0:
            buffer_samples = self.sample_from_training_buffer(num_buffer)
            buffer_samples = ensure_list(buffer_samples)
            mixed.extend(buffer_samples)

        shuffle(mixed)
        return batch_data(mixed)

    def sample_initial_data(self):
        """
        Run the sampler for num_init_samples, fill up the buffer.
        """
        log.info(
            f"Starting data generation of initial {self.hparams.num_init_samples} samples"
            + f" to populate training buffer with {self.sampler.num_mcmc_steps} MCMC steps"
            + f" and step size = {self.sampler.step_size}"
        )
        pos = self.sampler.sample(
            num_samples=self.hparams.num_init_samples, verbose=True
        )
        self.training_buffer.add(pos)
        self.num_generated += len(pos)
        self.training_buffer.save(self.buffer_save_path)

    def validate(self, state):
        valid_losses = []
        valid_dataloader = self.datamodule.valid_dataloader()

        with jax.default_device(self.train_device):
            jax.clear_caches()
            for graph in (pbar := tqdm(valid_dataloader, leave=False)):
                graph = jax.device_put(batch_data(graph), self.train_device)
                jax.block_until_ready(graph)
                valid_loss_step = eval_step(state, graph)
                valid_losses.append(valid_loss_step)
                pbar.set_description(f"Validation Loss: {valid_loss_step:06f}")

        return jnp.mean(jnp.array(valid_losses)).item()
