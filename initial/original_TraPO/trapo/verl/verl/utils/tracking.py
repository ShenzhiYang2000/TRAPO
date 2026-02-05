# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any
import os
os.environ["WANDB_MODE"] = "offline"


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None, log_file=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}
        self.log_file = log_file

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

            # # 新增：如果指定了日志文件，设置同时写入文件
            # if log_file:
            #     import logging
            #     from pathlib import Path
                
            #     # 确保目录存在
            #     Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                
            #     # 创建文件处理器
            #     file_handler = logging.FileHandler(log_file)
            #     file_handler.setLevel(logging.INFO)
            #     formatter = logging.Formatter('%(asctime)s - %(message)s')
            #     file_handler.setFormatter(formatter)
                
            #     # 添加到根日志记录器（这样所有print都会写入文件）
            #     root_logger = logging.getLogger()
            #     root_logger.addHandler(file_handler)
            #     root_logger.setLevel(logging.INFO)

                        # 新增：直接修改 LocalLogger 的 log 方法
            if log_file:
                import logging
                from pathlib import Path
                from datetime import datetime
                
                # 确保目录存在
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                
                # 保存原始 log 方法
                original_log = self.console_logger.log
                
                def new_log(data, step):
                    # 先调用原始方法（输出到控制台）
                    original_log(data, step)
                    
                    # 然后写入文件
                    try:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            message = f"Step {step}: {data}"
                            f.write(f"{timestamp} - {message}\n")
                    except Exception as e:
                        print(f"写入日志文件失败: {e}")
                
                # 替换 log 方法
                self.console_logger.log = new_log

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans
