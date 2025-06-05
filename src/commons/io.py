import os

import rootutils
from dotenv import load_dotenv
from omegaconf import OmegaConf


def read_yaml(yaml_path: str) -> dict:
    project_root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    env_filepath = os.path.join(project_root, ".env")
    load_dotenv(env_filepath, override=True)
    dict_config = OmegaConf.load(yaml_path)
    return OmegaConf.to_container(dict_config, resolve=True)
