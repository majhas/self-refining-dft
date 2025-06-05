import os
from collections import defaultdict
from datetime import datetime

from loguru import logger as log
from omegaconf import OmegaConf

import wandb


class WandBLogger:
    def __init__(
        self,
        project: str,
        entity: str,
        name: str = None,
        dir: str = None,
        id: str = None,
        log_step_frequency: int = 100,
        log_epoch_frequency: int = 1,
        on_epoch: bool = False,
        on_step: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.log_step_frequency = log_step_frequency
        self.log_epoch_frequency = log_epoch_frequency
        self.on_epoch = on_epoch
        self.on_step = on_step
        self.running_data_dict_step = defaultdict(list)
        self.running_data_dict_epoch = defaultdict(list)
        self.run = wandb.init(
            name=name,
            project=project,
            entity=entity,
            dir=dir,
            id=id,
            resume="allow",
        )

        if on_epoch:
            wandb.define_metric(name="train/loss", step_metric="epoch")
            wandb.define_metric(name="valid/loss", step_metric="epoch")

    def log(
        self,
        data_dict: dict = None,
        step: int = None,
        epoch: int = None,
        force_log: bool = False,
    ):
        self.add(data_dict=data_dict, on_step=(step is not None))
        if (self.on_step and step is not None) and (
            (step % self.log_step_frequency == 0) or force_log
        ):
            assert step is not None, "step cannot be None"
            self._log(step=step)

        elif (self.on_epoch and epoch is not None) and (
            (epoch % self.log_epoch_frequency == 0) or force_log
        ):
            assert epoch is not None, "epoch cannot be None"
            self._log(epoch=epoch)

    def log_metrics(self, data_dict: dict):
        self.run.log(data=data_dict)

    def log_hyperparameters(self, data_dict: dict):
        self.run.config.update(data_dict, allow_val_change=True)

    def clear(self, on_step: bool = False, on_epoch: bool = False):
        if on_step:
            self.running_data_dict_step = defaultdict(list)

        if on_epoch:
            self.running_data_dict_epoch = defaultdict(list)

    def add(self, data_dict: dict, on_step: bool = False):
        if data_dict is None:
            return

        for key, value in data_dict.items():
            if self.on_epoch:
                self.running_data_dict_epoch[key].append(value)

                if on_step:
                    key = "_".join((key, "step"))

            self.running_data_dict_step[key].append(value)

    def _log(self, step=None, epoch=None):
        def mean(value):
            return sum(value) / len(value)

        if step is not None:
            _step_in_dict = [k for k in self.running_data_dict_step if "step" in k][0]
            if isinstance(_step_in_dict, list):
                _step_in_dict = _step_in_dict[-1]

            data_dict = {
                key: mean(value) for key, value in self.running_data_dict_step.items()
            }
            data_dict["step"] = _step_in_dict
            self.run.log(data=data_dict, step=step)
            self.clear(on_step=True)

        if epoch is not None:
            data_dict = {
                key: mean(value) for key, value in self.running_data_dict_epoch.items()
            }
            data_dict["epoch"] = epoch
            self.run.log(data=data_dict, step=step)
            self.clear(on_epoch=True)

    def log_table(self, columns, data):
        table = wandb.Table(columns=columns, data=data)
        self.run.log({"Metrics": table})

def get_log_dir():
    """
    Check if LOG_DIR is env variable is set
    else use logs/ directory
    """
    log_dir = os.environ.get("LOG_DIR")
    if log_dir is None:
        log_dir = "logs/"

    return log_dir


def setup_log_dir(task_name):
    """Sets log directory for a given task
    and then moves to that directory
    """
    log_dir = get_log_dir()
    # use time to create unique log diedge_indexrectory
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, task_name, "runs", f"{run_name}")
    log.info(f"Log directory: {log_dir}")

    # create log directory
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)


def log_hyperparameters(logger: WandBLogger, object_dict: dict) -> None:
    """Controls which config parts are saved by wandb logger.

    Additionally saves:
    - Number of model parameters
    """

    if logger is None:
        return

    cfg = object_dict["cfg"]
    serializable_cfg = OmegaConf.to_container(cfg, resolve=True)

    # send hparams to all loggers
    logger.log_hyperparameters(serializable_cfg)
