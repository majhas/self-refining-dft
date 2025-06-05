import logging
import os

import rootutils

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
logging.getLogger("absl").setLevel(logging.ERROR)

import hydra
import jax
import numpy as np
from loguru import logger as log
from dotenv import load_dotenv
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

rootutils.setup_root(__file__, pythonpath=True)
load_dotenv(override=True)

from src.commons.logger import log_hyperparameters


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")


    log.info(f"PROJECT ROOT: {os.environ['PROJECT_ROOT']}")
    log.info(f"SCRATCH DIR: {os.environ['SCRATCH_DIR']}")
    
    debug = cfg.debug
    seed = cfg.seed if cfg.seed is not None else 42
    wandbid = cfg.logger.get("id", None)
    if wandbid is None:
        wandbid = str(np.random.randint(int(1e7), int(1e8)))
        cfg.logger.id = wandbid

    task_name = "debug" if debug else cfg.task_name
    assert task_name is not None, "Please set task_name in your config!"
    key = jax.random.PRNGKey(seed)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    if debug or cfg.no_logger:
        logger = None

    # log config and model to wandb
    log_hyperparameters(logger=logger, object_dict={"cfg": cfg, "model": model})

    ckpt_dir = cfg.get("ckpt_dir", None)
    if ckpt_dir is not None:
        ckpt_dir = to_absolute_path(ckpt_dir)

    trainer = instantiate(
        cfg.trainer, key=key, datamodule=datamodule, logger=logger, ckpt_dir=ckpt_dir
    )

    trainer.run_training()


if __name__ == "__main__":
    main()
