import os
import torch
import pytorch_lightning as pl
from loguru import logger
from .data.dataset import get_dataloaders
from .modeling.model import Model
from .training.engine import Engine
from .utils import util_system, util_implement
from .utils.decorators import *

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))


def _build_optimizer_parameter_groups(model):
    """Keep tagged parameters and learned controls free from AdamW decay."""
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_tagged_no_decay = bool(getattr(parameter, "_no_weight_decay", False))
        is_split_weight = ".split_shared_fusions." in name and name.endswith(".split_logit")
        is_adapter_scale = name.endswith(".spk_split.adapter_gammas")
        is_cpcm_scale = ".cpc_mamba." in name and name.endswith(".raw_scale")
        if (
            is_tagged_no_decay
            or is_split_weight
            or is_adapter_scale
            or is_cpcm_scale
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    if not no_decay_params:
        return decay_params
    return [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def _select_dataset_profile(config, data_type):
    dataset_config = config["dataset"]
    profiles = dataset_config.get("profiles", {})
    data_type = data_type or dataset_config.get("data_type", "clean")

    aliases = {
        "wsj0": "clean",
        "wsj0-2mix": "clean",
        "libri": "libri2mix",
        "librimix": "libri2mix",
        "wham": "noisy",
        "whamr": "noisy_reverb",
        "reverb": "noisy_reverb",
    }
    data_type = aliases.get(data_type, data_type)

    if not profiles:
        dataset_config["data_type"] = data_type
        return data_type

    if data_type not in profiles:
        raise ValueError(f"Unsupported DM dataset type: {data_type}. Choose from {list(profiles)}")

    shared_keys = ("max_len", "sampling_rate")
    selected = {key: dataset_config[key] for key in shared_keys}
    selected.update(profiles[data_type])
    selected["data_type"] = data_type
    if not os.path.isabs(selected["scp_dir"]):
        selected["scp_dir"] = os.path.join(PROJECT_DIR, selected["scp_dir"])
    if not os.path.isabs(selected["output_dir"]):
        selected["output_dir"] = os.path.join(PROJECT_DIR, selected["output_dir"])
    config["dataset"] = selected
    return data_type


def _setup_logger(dataset_config):
    log_dir = os.path.join(dataset_config["output_dir"], "logs")
    log_file_path = os.path.join(log_dir, "system_log.log")
    os.makedirs(log_dir, exist_ok=True)
    is_first_run = not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0

    logger.remove()
    logger.add(
        log_file_path,
        level="INFO",
        mode="a",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
    )
    return is_first_run

# 注意：这里去掉了 @logger_wraps()，防止它把 model 和 optimizer 打印出来
def main(args):
    ''' Build Setting '''
    yaml_path = os.path.join(PACKAGE_DIR, "configs.yaml")
    yaml_dict = util_system.parse_yaml(yaml_path)
    config = yaml_dict["config"]
    data_type = _select_dataset_profile(config, getattr(args, "dm_data", None))
    is_first_run = _setup_logger(config["dataset"])
    seed = int(config.get("seed", 2026))
    pl.seed_everything(seed, workers=False)
    
    # 只有第一次运行才写配置
    if is_first_run:
        logger.info(f"DM dataset type: {data_type}")
        logger.info(f"Full Configuration: {config}")
    else:
        logger.info(f"--- Resuming Training: Configuration logging skipped ({data_type}) ---")

    dataloaders = get_dataloaders(
        args, config["dataset"], config["dataloader"], seed=seed
    )
    
    ''' Build Model '''
    model = Model(**config["model"])
    pl.seed_everything(seed, workers=False)

    ''' Build Engine '''
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')
    
    criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
    optimizer_parameters = _build_optimizer_parameter_groups(model)
    optimizers = util_implement.OptimizerFactory(config["optimizer"], optimizer_parameters).get_optimizers()
    schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()
    
    # 初始化 Engine (同样确保 Engine 类内部不被装饰器干扰输出参数)
    engine = Engine(args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device)
    engine.run()
