import argparse
import torch
import os
import sys
import logging
from torchvision import transforms
import functools
import shutil
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode
import importlib.util


def load_module_from_path(file_path, module_name='dynamic_module'):
    """Load a module dynamically from a given file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def SetDirectory(config, custom_name=""):
    """Set up experiment directories for checkpoints, logs, results, etc."""
    parent_path = os.path.join(config.OUTPUT_DIR, config.EXP_NAME)
    time = datetime.now().strftime("%m-%d-%H-%M-%S")
    out_dir = os.path.join(parent_path, f"{custom_name}{time}")

    # Create subdirectories
    ckpt_path = check_dir(os.path.join(out_dir, config.CKPT))  # Checkpoints
    log_path = check_dir(os.path.join(out_dir, config.LOG))  # Logs
    result_path = check_dir(os.path.join(out_dir, config.RESULT))  # Predictions
    tb_path = check_dir(os.path.join(out_dir, config.TB))  # TensorBoard logs

    # Create config file if it doesn't exist
    check_file(out_dir, "config.yml")
    save_cfg_to_yaml(config, os.path.join(out_dir, "config.yml"))  # For self-check

    return out_dir, ckpt_path, result_path, tb_path, log_path


def cfg_node_to_dict(cfg_node):
    """Convert YACS CfgNode to regular dictionary"""
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CfgNode):
            cfg_dict[key] = cfg_node_to_dict(value)
        else:
            cfg_dict[key] = value
    return cfg_dict


def save_cfg_to_yaml(cfg, filepath):
    """Save configuration to YAML file"""
    cfg_dict = cfg_node_to_dict(cfg)
    with open(filepath, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def dict2namespace(config):
    """Convert dictionary to argparse Namespace"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def check_dir(dir_path):
    """Create directory if it doesn't exist (handles multi-GPU race conditions)"""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass  # Directory already exists
    return dir_path


def check_file(dir_path, filename):
    """Create empty file if it doesn't exist"""
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            print(f"{filename} created successfully.")


def setup_logger(name: str, log_dir: str, rank: int, filename: str):
    """Set up a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    formatter = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_main_process_logger(cfg, accelerator, log_dir, filename="main_log.txt"):
    """Set up logger for main process (suppress for other processes)"""
    name = cfg.EXP_NAME

    # Non-main processes only log errors
    if not accelerator.is_main_process:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create log directory if needed
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    # File handler
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class DummyWriter:
    """Dummy TensorBoard writer for non-main processes"""

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def setup_main_process_writer(cfg, accelerator, tb_path, filename="main_log.txt"):
    """Set up TensorBoard writer (dummy for non-main processes)"""
    if accelerator.is_main_process:
        writer = SummaryWriter(tb_path)
    else:
        writer = DummyWriter()
    return writer


def reverse_normalize(normalized_image):
    """Reverse ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    inv_normalize = transforms.Normalize(
        (-mean / std).tolist(),
        (1.0 / std).tolist()
    )
    return inv_normalize(normalized_image)


def Tensor2PIL(img_tensor):
    """Convert tensor to PIL Image"""
    return transforms.ToPILImage()(img_tensor)


def cal_param(model):
    """Calculate model parameters (total, trainable, non-trainable) in millions"""
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        param_count = np.prod(param.size())
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

    return (
        total_params / 1e6,
        trainable_params / 1e6,
        non_trainable_params / 1e6
    )


def l2_norm(hidden_states):
    """Calculate L2 norm along the last dimension"""
    return hidden_states.norm(p=2, dim=-1, keepdim=True)