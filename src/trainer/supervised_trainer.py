import traceback

import jax.numpy as jnp
from flax.training import orbax_utils
from loguru import logger as log
from tqdm import trange

from src.commons.graph import batch_data
from src.trainer.base_trainer import BaseTrainer, train_step
from src.trainer.utils import get_lr_from_opt


class SupervisedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_training(self):
        self.state = self.initialize_model()
        start_step = self.state.step

        log.info("Starting training loop")
        valid_loss = jnp.inf

        try:
            for step in (pbar := trange(start_step, self.hparams.num_iterations + 1)):
                if (step == start_step) or (step % self.hparams.num_recycle_steps == 0):
                    graph = next(self.train_data_iter, None)
                    if graph is None:
                        self.train_data_iter = self.init_dataloader(
                            size=self.hparams.num_data_samples,
                            batch_size=self.hparams.batch_size,
                        )
                        graph = next(self.train_data_iter)
                    graph = batch_data(graph)

                self.state, loss, grad_norm = train_step(self.state, graph)

                if self.logger:
                    learning_rate = get_lr_from_opt(self.state)
                    log_dict = {
                        "train/loss": loss,
                        "train/grad_norm": grad_norm,
                        "train/step": step,
                        "learning-rate": learning_rate,
                    }

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
            metrics = {}
        else:
            checkpoint_manager = self.checkpoint_manager
            metrics = {"valid_loss": loss}

        self.state = self.state.replace(key=self.key)

        save_args = orbax_utils.save_args_from_target(self.state)
        checkpoint_manager.save(
            step,
            self.state,
            save_kwargs={"save_args": save_args},
            metrics=metrics,
            force=True,
        )
