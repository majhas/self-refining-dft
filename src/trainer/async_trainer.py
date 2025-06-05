import threading
import traceback

import jax
import jax.numpy as jnp
from flax.training import orbax_utils
from loguru import logger as log
from tqdm import trange

from src.trainer.base_trainer import BaseSelfRefiningTrainer, train_step
from src.trainer.utils import TrainState, get_lr_from_opt


class AsyncSelfRefiningTrainer(BaseSelfRefiningTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.training_buffer_lock = threading.Lock()
        self.sample_data_lock = threading.Lock()

    def initialize_state(self):
        params = self.initialize_params()
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            key=jax.device_put(self.key, self.train_device),
            step_size=getattr(self, "step_size", 1e-3),
            num_generated=getattr(self, "num_generated", 0),
        )
        return state

    def sample_from_training_buffer(self, batch_size=32):
        with self.training_buffer_lock:
            return super().sample_from_training_buffer(batch_size)

    def sample_mixed_batch(self, batch_size=32, use_sampler_iter=False):
        with self.sample_data_lock:
            return super().sample_mixed_batch(batch_size, use_sampler_iter)

    def sync(self):
        with self.training_buffer_lock:
            for pos in self.sampler.buffer:
                self.training_buffer.add(pos)

            # Reset buffer after adding them to train buffer
            self.num_generated += len(self.sampler.buffer)
            self.sampler.buffer.empty()

        # Synchronize the data generation model with
        # updated parameters of the training model
        self.sampler.update_params(self.state.params)

    def run_training(self):
        with jax.default_device(self.train_device):
            self.state = self.initialize_model()

        log.info("Starting training loop")

        if self.state.step < self.hparams.num_pretrain_steps:
            log.info(
                f"Pretraining model for first {self.hparams.num_pretrain_steps} iterations"
            )

        start_step = self.state.step
        valid_loss = jnp.inf
        started_sampler = False

        try:
            for step in (pbar := trange(start_step, self.hparams.num_iterations + 1)):
                if step >= self.hparams.num_pretrain_steps and not started_sampler:
                    self.sampler.update_params(self.state.params)
                    if step == self.hparams.num_pretrain_steps:
                        log.info(
                            "Done Pretraining. "
                            f"Generating initial {self.hparams.num_init_samples} samples"
                        )
                        self.sample_initial_data()

                    self.sampler.start()
                    started_sampler = True

                if self.sampler.check_sync():
                    self.sync()
                    self.sampler.signal_completed_sync()

                if (step == start_step) or (step % self.hparams.num_recycle_steps == 0):
                    graph = self.sample_mixed_batch(self.hparams.batch_size)

                self.state, loss, grad_norm = train_step(self.state, graph)

                if self.logger:
                    learning_rate = get_lr_from_opt(self.state)
                    log_dict = {
                        "train/loss": loss,
                        "train/grad_norm": grad_norm,
                        "train/step": step,
                        "train/num_generated": int(self.num_generated),
                        "train/step_size": self.sampler.step_size,
                        "learning-rate": learning_rate,
                    }

                    if self.sampler.acceptance_rate is not None:
                        log_dict["train/acceptance_rate"] = self.sampler.acceptance_rate

                    self.logger.log(log_dict, step=step)

                if (step % self.hparams.save_every == 0) and step != start_step:
                    self.checkpoint(step, latest=True)

                if (step % self.hparams.eval_freq == 0) and step != start_step:
                    valid_loss = self.validate(self.state)
                    self.checkpoint(step, loss=valid_loss, latest=False)

                    # Logging
                    if self.logger:
                        self.logger.log(
                            {"valid/loss": valid_loss, "step": step},
                            step=step,
                            force_log=True,
                        )

                pbar.set_description(
                    f"Step {step}: Loss = {loss:.8f}, Valid Loss = {valid_loss:.8f},"
                    + f" Training Buffer size = {len(self.training_buffer)},"
                    + f" Data Gen Buffer size = {len(self.sampler.buffer)}"
                    + f" Num Generated = {int(self.num_generated)}"
                )

        except Exception as e:
            log.error(f"Training loop error: {e}")
            log.error("Full traceback:\n" + traceback.format_exc())
        finally:
            if self.sampler.thread.is_alive():
                self.sampler.stop()

            self.checkpoint_manager.close()
            self.latest_checkpoint_manager.close()

    def checkpoint(self, step, loss=None, latest=False):
        if latest:
            checkpoint_manager = self.latest_checkpoint_manager
            self.training_buffer.save(filepath=self.buffer_save_path)
            metrics = {}
        else:
            checkpoint_manager = self.checkpoint_manager
            metrics = {"valid_loss": loss}

        self.state = self.state.replace(
            key=self.key,
            step_size=self.sampler.step_size,
            num_generated=self.num_generated,
        )
        save_args = orbax_utils.save_args_from_target(self.state)
        checkpoint_manager.save(
            step,
            self.state,
            save_kwargs={"save_args": save_args},
            metrics=metrics,
            force=True,
        )
