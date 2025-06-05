import traceback

import jax
import jax.numpy as jnp
import optax
from flax.training import orbax_utils
from loguru import logger as log
from tqdm import trange

from src.commons.graph import batch_data
from src.trainer.base_trainer import BaseSelfRefiningTrainer, train_step
from src.trainer.utils import TrainState, get_lr_from_opt


class SelfRefiningTrainer(BaseSelfRefiningTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pretrain(self):
        """Pretrain the model using only pre-collected dataset samples."""
        log.info(
            f"Pretraining model for {self.num_pretrain_steps} steps "
            f"using {self.num_pretrain_steps} samples."
        )

        data_iter = self.init_dataloader(
            size=self.num_data_samples, batch_size=self.train_batch_size
        )

        lr_args = self.config.get("lr_scheduler_args", {})
        lr_args.setdefault("warmup_steps", self.num_pretrain_steps // 10)
        lr_args.setdefault("decay_steps", self.num_pretrain_steps)

        lr_schedule = optax.warmup_cosine_decay_schedule(**lr_args)
        pretrain_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr_schedule)

        # Init pretrain state with existing params
        pretrain_state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.state.params,
            tx=pretrain_tx,
            key=0,  # dummy
            step_size=0,  # dummy
        )

        for step in (pbar := trange(self.num_pretrain_steps, desc="Pretraining")):
            if step % self.num_recycle_steps == 0:
                batch = next(data_iter, None)
                if batch is None:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

            graph = batch_data(batch)
            pretrain_state, loss, _ = train_step(pretrain_state, graph)

            if self.logger:
                self.logger.log(
                    {
                        "Pretrain/loss": loss,
                        "Pretrain/lr": lr_schedule(step),
                        "Pretrain/step": step,
                    },
                    step=step,
                )

            pbar.set_description(f"Step {step}: Loss = {loss:.4f}")

        self.state = self.state.replace(params=pretrain_state.params, step=1)

        if self.latest_checkpoint_manager:
            save_args = orbax_utils.save_args_from_target(self.state)
            self.latest_checkpoint_manager.save(
                step=1,
                items=self.state,
                save_kwargs={"save_args": save_args},
                force=True,
            )

    def sample_and_add(self):
        self.sampler.params = self.state.params
        pos = self.sampler.sample()
        self.training_buffer.add(pos)

    def run_training(self):
        params = self.initialize_model()
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            key=jax.device_put(self.key, self.train_device),
            step_size=getattr(self, "step_size", 1e-3),
            num_generated=getattr(self, "num_generated", 0),
        )
        self.state = self.restore_from_checkpoint(self.state)
        start_step = self.state.step
        if start_step == 0 and self.num_pretrain_steps > 0:
            log.info("Pretraining model before starting main training loop")
            self.pretrain()

        self.sampler.params = self.state.params
        self.sample_initial_data()

        log.info("Starting training loop")
        valid_loss = jnp.inf

        try:
            for step in (pbar := trange(start_step, self.num_iterations + 1)):
                if (step == start_step) or (step % self.num_recycle_steps == 0):
                    self.sample_and_add()
                    graph = self.sample_mixed_batch(self.train_batch_size)

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

                    self.logger.log(log_dict, step=step + self.num_pretrain_steps)

                if (step % self.save_every == 0) and step != 0:
                    self.checkpoint(step, latest=True)

                if (step % self.eval_freq == 0) and step != 0:
                    valid_loss = self.validate(self.state)
                    self.checkpoint(step, loss=valid_loss, latest=False)

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
