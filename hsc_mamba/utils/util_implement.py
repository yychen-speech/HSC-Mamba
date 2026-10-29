import torch
import importlib
from dataclasses import dataclass
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from .decorators import *


@dataclass(slots=True)
class BaseFactory:
    config: dict

    def load_class(self, category: str, lib_check: Type[Any], name: str) -> Type[Any]:
        """
        Loading Classes from a Module
        Modified: 增加了导入路径的鲁棒性，优先检查 torch 库，然后是 utils.implements 子模块，最后尝试直接导入 (针对 criterions.py 在根目录的情况)
        """
        # 1. 优先检查 lib_check (如 torch.nn, torch.optim)
        if hasattr(lib_check, name):
            return getattr(lib_check, name)
        
        # 2. 尝试从 utils.implements.{category} 导入 (这是原项目的默认结构)
        try:
            return getattr(importlib.import_module(f"hsc_mamba.utils.implements.{category}"), name)
        except (ImportError, AttributeError):
            pass # 继续尝试其他路径

        # 3. [新增] 尝试从直接模块导入 (例如直接 import criterions)
        # 这对应如果你的 criterions.py 文件放在了根目录或者 utils 目录下而不在 implements 里的情况
        try:
            # 尝试导入 category 名字同名的模块 (e.g., import criterions)
            return getattr(importlib.import_module(category), name)
        except (ImportError, AttributeError) as e:
            # 如果都找不到，则报错
            logger.error(f"Class '{name}' not found in 'hsc_mamba.utils.implements.{category}' or '{category}'. Error: {e}")
            raise

    def create_instance(self, cls: Type[Any], name: str, target: Any = None) -> Any:
        """Create Instance"""
        arg_dict = self.config.get(name, {}).copy() # 使用 .copy() 防止修改原始配置
        
        # [安全防御] 如果 yaml 里意外写了 device, 把它删掉，避免和代码里传入的 device 冲突导致 "multiple values" 错误
        if 'device' in arg_dict:
            arg_dict.pop('device')

        logger.info(f'Creating {cls.__name__} instance with args: {arg_dict}')
        
        # 如果传入了 target (例如 device, parameters 等), 把它作为第一个参数 (positional argument)
        if target is not None:
            return cls(target, **arg_dict)
        else:
            return cls(**arg_dict)

    def get_instances(self, category: str, lib_check: Type[Any], instance_creator: Callable) -> List[Any]:
        """Create Instance List"""
        # 遍历 yaml 配置中 name 列表里的所有名称，依次实例化
        return [instance_creator(self.load_class(category, lib_check, name), name) for name in self.config["name"]]


@logger_wraps()
@dataclass(slots=True)
class CriterionFactory(BaseFactory):
    device: torch.device
    
    def get_criterions(self) -> List[Type[Any]]:
        # 加载 criterions 模块下的类，并将 self.device 作为第一个参数传入
        # 这会匹配 PIT_HybridLoss(device, ...) 的签名
        return self.get_instances("criterions", torch.nn, lambda cls, name: self.create_instance(cls, name, self.device))


@logger_wraps()
@dataclass(slots=True)
class OptimizerFactory(BaseFactory):
    parameters_policy: Any

    def get_optimizers(self) -> List[Type[Any]]:
        return self.get_instances("optimizers", torch.optim, lambda cls, name: self.create_instance(cls, name, self.parameters_policy))


@logger_wraps()
@dataclass(slots=True)
class SchedulerFactory(BaseFactory):
    optimizers: List[Any]

    def get_schedulers(self) -> List[Type[Any]]:
        optimizers = self.optimizers
        # 如果 yaml 里配了多个 scheduler (比如 warm up + reduce on plateau)，但只有一个 optimizer
        # 则把 optimizer 复制多份来匹配
        if len(self.optimizers) == 1 and len(self.config["name"]) > 1:
            optimizers = self.optimizers * len(self.config["name"]) 

        return [self.create_instance(self.load_class("schedulers", torch.optim.lr_scheduler, name), name, optimizer) 
                for name, optimizer in zip(self.config["name"], optimizers)]
