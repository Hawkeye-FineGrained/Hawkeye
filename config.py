import argparse
from yacs.config import CfgNode as CN


def setup_config():
    """
    Only load from one single yaml file.
    :return: CfgNode
    """
    arg = _parse_args()
    if arg.config is not None:
        # cfg.merge_from_file(arg.config)
        with open(arg.config) as f:
            cfg = CN.load_cfg(f)
    else:
        cfg = _get_default_config()
    cfg.freeze()
    return cfg


def _parse_args():
    parser = argparse.ArgumentParser(description='Hakweye')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    arg = parser.parse_args()
    return arg


def _get_default_config():
    BASE_CONFIG_PATH = 'configs/Baseline.yaml'
    with open(BASE_CONFIG_PATH) as f:
        cfg = CN.load_cfg(f)
    return cfg
