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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import copy

from itertools import zip_longest

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean


# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_split':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

def check_length(new_dict, return_max_length = False, return_min_length = False):
    if not new_dict:
        return None
    # Calculate the maximum length of all values
    max_length = -1
    min_length = 1000000
    for value in new_dict.values():
        if hasattr(value, 'shape') and len(value.shape) > 0:
            # For a tensor, use the first dimension of the shape as the length.
            current_length = value.shape[0]
        elif hasattr(value, '__len__'):
            # For other objects of measurable length
            current_length = len(value)
        else:
            # Objects without a length property default to a length of 1
            current_length = 1
        
        if current_length > max_length:
            max_length = current_length
        if current_length < min_length:
            min_length = current_length
    
    if return_max_length and return_min_length:
        return max_length, min_length
    elif return_max_length:
        return max_length
    elif return_min_length:
        return min_length
    else:
        return None


class MIXRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

        self.label_passrate_list = {}
        self.unlabel_passrate_list = {}
        self.real_unlabel_passrate_list = {}

        self.old_label_passrate_list = {}
        self.old_unlabel_passrate_list = {}
        self.old_real_unlabel_passrate_list = {}

        self.new_epoch_flag = True
        self.start_epoch_flag = True

        self.ref_passrate_list = {}
        self.old_ref_passrate_list = {}

        self.max_length = -1
        self.old_max_length = -1
        self.min_length = 1000000

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_split':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        self.unlabel_train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.unlabel_train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        
        # breakpoint()
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
            unlabel_sampler = ResumableRandomSampler(data_source=self.unlabel_train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
            unlabel_sampler = SequentialSampler(data_source=self.unlabel_train_dataset)
        


        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.unlabel_train_dataloader = DataLoader(dataset=self.unlabel_train_dataset,
                                           batch_size=self.config.data.unlabel_train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=unlabel_sampler)

                                           
        
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.unlabel_train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of unlabeled train dataloader: {len(self.unlabel_train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def find_top_related_cosine(self, query_dict, database_dict, top_k_ratio=0.1, aggregate_method='mean'):
        
        if isinstance(query_dict, dict):
            query_tensors = list(query_dict.values())
            query_keys = list(query_dict.keys())
            query = torch.stack(query_tensors, dim=0)
        else:
            query = query_dict
            query_keys = None
        
        if isinstance(database_dict, dict):
            database_tensors = list(database_dict.values())
            database_keys = list(database_dict.keys())
            database = torch.stack(database_tensors, dim=0)
        else:
            database = database_dict
            database_keys = None
        
        if query.dim() != 2 or database.dim() != 2:
            raise ValueError("tensor must be 2d (samples, features)")
        
        if query.shape[1] != database.shape[1]:

            raise ValueError(f"wrong feature dim: query={query.shape[1]}, database={database.shape[1]}")
        
        a, b = query.shape
        c = database.shape[0]
        k = int(c * top_k_ratio)  

        k_1 = int(c * 0.1) 
        k_2 = int(c * 0.2)  
        k_3 = int(c * 0.3) 
        k_4 = int(c * 0.4)  
        k_5 = int(c * 0.5)  
        k_6 = int(c * 0.6)  
        k_7 = int(c * 0.7) 
        k_8 = int(c * 0.8) 
        k_9 = int(c * 0.9)  
        
        if k == 0:
            k = 1  # at least 1
        
        query_norm = (query - query.mean(dim=1, keepdim=True)) / (query.std(dim=1, keepdim=True) + 1e-8)
        db_norm = (database - database.mean(dim=1, keepdim=True)) / (database.std(dim=1, keepdim=True) + 1e-8)
        
        # query_norm: (a, 1, b), db_norm: (1, c, b) -> compute cos_sim -> (a, c)
        cosine_sim = F.cosine_similarity(query_norm.unsqueeze(1), db_norm.unsqueeze(0), dim=2)
    
        if aggregate_method == 'mean':
            aggregate_scores = cosine_sim.mean(dim=0)  
        elif aggregate_method == 'max':
            aggregate_scores = cosine_sim.max(dim=0).values  
        else:
            raise ValueError("aggregate_method must be 'mean' or 'max'")
        
        
        top_scores, top_indices = torch.topk(aggregate_scores, k=k)
        top_0_1_scores, top_0_1_indices = torch.topk(aggregate_scores, k=k_1)
        top_0_2_scores, top_0_2_indices = torch.topk(aggregate_scores, k=k_2)
        top_0_3_scores, top_0_3_indices = torch.topk(aggregate_scores, k=k_3)
        top_0_4_scores, top_0_4_indices = torch.topk(aggregate_scores, k=k_4)
        top_0_5_scores, top_0_5_indices = torch.topk(aggregate_scores, k=k_5)
        top_0_6_scores, top_0_6_indices = torch.topk(aggregate_scores, k=k_6)
        top_0_7_scores, top_0_7_indices = torch.topk(aggregate_scores, k=k_7)
        top_0_8_scores, top_0_8_indices = torch.topk(aggregate_scores, k=k_8)
        top_0_9_scores, top_0_9_indices = torch.topk(aggregate_scores, k=k_9)

        top_scores, top_indices = torch.topk(aggregate_scores, k=k)
        mask = aggregate_scores > self.config.actor_rollout_ref.tau
        filtered_indices = torch.where(mask)[0]

        combined_indices = torch.cat([top_indices, filtered_indices])
        final_indices = torch.unique(combined_indices)
        final_scores = aggregate_scores[final_indices]
        
        if database_keys is not None:
            top_keys = [database_keys[i] for i in top_indices.cpu().numpy()]
            top_0_1_keys = [database_keys[i] for i in top_0_1_indices.cpu().numpy()]
            top_0_2_keys = [database_keys[i] for i in top_0_2_indices.cpu().numpy()]
            top_0_3_keys = [database_keys[i] for i in top_0_3_indices.cpu().numpy()]
            top_0_4_keys = [database_keys[i] for i in top_0_4_indices.cpu().numpy()]
            top_0_5_keys = [database_keys[i] for i in top_0_5_indices.cpu().numpy()]
            top_0_6_keys = [database_keys[i] for i in top_0_6_indices.cpu().numpy()]
            top_0_7_keys = [database_keys[i] for i in top_0_7_indices.cpu().numpy()]
            top_0_8_keys = [database_keys[i] for i in top_0_8_indices.cpu().numpy()]
            top_0_9_keys = [database_keys[i] for i in top_0_9_indices.cpu().numpy()]
            final_keys = [database_keys[i] for i in final_indices.cpu().numpy()]
            
            return top_indices, top_scores, top_keys, \
            top_0_1_scores, top_0_2_scores, top_0_3_scores, top_0_4_scores, top_0_5_scores, top_0_6_scores, top_0_7_scores, top_0_8_scores, top_0_9_scores, final_scores, \
            top_0_1_keys, top_0_2_keys, top_0_3_keys, top_0_4_keys, top_0_5_keys, top_0_6_keys, top_0_7_keys, top_0_8_keys, top_0_9_keys, final_keys
        else:
            return top_indices, top_scores, None


    def copy_dict_with_validation(self, original_dict, prefix="label", filter_by_max_length=False):

        for key, value in original_dict.items():
            new_key = f"{prefix}_{key}"
            
            if new_key in self.ref_passrate_list:
                self.old_ref_passrate_list = copy.deepcopy(self.ref_passrate_list)
                self.ref_passrate_list = {}
                self.ref_passrate_list[new_key] = value
            else:
                # 键不存在，直接添加
                self.ref_passrate_list[new_key] = value

    def fit_supervised(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          log_file = self.config.actor_rollout_ref.output_log_path)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch, is_labeled=True) # [bsz, l], only the last valid token has reward

                        batch.batch['token_level_scores'] = reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)
                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch.meta_info['is_labeled'] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return



    def fit_unsupervised(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          log_file = self.config.actor_rollout_ref.output_log_path)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch, is_labeled=False) # [bsz, l], only the last valid token has reward

                        batch.batch['token_level_scores'] = reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)
                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch.meta_info['is_labeled'] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return



    def fit_semi_supervised(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          log_file = self.config.actor_rollout_ref.output_log_path)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict, unlabel_batch_dict in zip(self.train_dataloader, self.unlabel_train_dataloader):
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                unlabel_batch: DataProto = DataProto.from_single_dict(unlabel_batch_dict)

                metrics = {}
                timing_raw = {}

                # unlabel_metrics = {}
                # unlabel_timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                unlabel_gen_batch = unlabel_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                unlabel_gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        unlabel_gen_batch_output = self.actor_rollout_wg.generate_sequences(unlabel_gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    unlabel_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(unlabel_batch.batch))],
                                                             dtype=object)
                    unlabel_batch = unlabel_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    unlabel_batch = unlabel_batch.union(unlabel_gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))
                        metrics['ul_batch/avg_prefix_ratio'] = float(np.mean(unlabel_gen_batch_output.meta_info['prefix_ratios']))
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                            unlabel_values = self.critic_wg.compute_values(unlabel_batch)
                            unlabel_batch = unlabel_batch.union(unlabel_values)


                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                            unlabel_reward_tensor = self.rm_wg.compute_rm_score(unlabel_batch)
                            unlabel_batch = batch.union(unlabel_reward_tensor)

                        reward_tensor = self.reward_fn(batch, is_labeled = self.config.actor_rollout_ref.labeled_is_label) # [bsz, l], only the last valid token has reward

                        unlabel_reward_tensor = self.reward_fn(unlabel_batch, is_labeled = self.config.actor_rollout_ref.unlabeled_is_label) # [bsz, l], only the last valid token has reward
                        # unlabel_reward_tensor = self.reward_fn(unlabel_batch, is_labeled = True) # [bsz, l], only the last valid token has reward

                        real_unlabel_reward_tensor = self.reward_fn(unlabel_batch, is_labeled = True) # [bsz, l], only the last valid token has reward


                        for idx, row in enumerate(batch.batch['prefix_mask']):
                            if row.any() and not reward_tensor[idx].any():
                                reward_tensor[idx, len(reward_tensor[idx])-1] = True

                        for idx, row in enumerate(unlabel_batch.batch['prefix_mask']):
                            if row.any() and not unlabel_reward_tensor[idx].any():
                                unlabel_reward_tensor[idx, len(unlabel_reward_tensor[idx])-1] = True

                        batch.batch['token_level_scores'] = reward_tensor
                        unlabel_batch.batch['token_level_scores'] = unlabel_reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        unlabel_uids = unlabel_batch.non_tensor_batch['uid']
                        unlabel_unique_uids = np.unique(unlabel_uids)
                        unlabel_valid_mask = torch.ones(len(unlabel_uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0

                        label_question_accuracy_rate = torch.zeros([reward_tensor.shape[0],]).to(reward_tensor.device)

                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            label_accuracy_rate = uid_rewards.sum(0) / uid_rewards.shape[0]
                            label_question_accuracy_rate[uid_mask] = label_accuracy_rate

                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1


                        label_mean_accuracy_rate = torch.mean(label_question_accuracy_rate, dim=0)
                        metrics['batch/mean_accuracy_rate'] = label_mean_accuracy_rate.item()

                        batch.batch['accuracy_rate'] = label_question_accuracy_rate

                        current_label_passrate_list = {}
                        current_index = []
                        tmp_label_question_accuracy_rate = label_question_accuracy_rate.unsqueeze(1)
                        for i, extra_info in enumerate(batch.non_tensor_batch['extra_info']):
                            if extra_info['index'] in current_index:
                                continue
                            current_index.append(extra_info['index'])
                            if extra_info['index'] not in self.label_passrate_list.keys():
                                self.label_passrate_list[extra_info['index']] = tmp_label_question_accuracy_rate[i]
                                current_label_passrate_list[extra_info['index']] = tmp_label_question_accuracy_rate[i]
                            else:
                                current_label_passrate_list[extra_info['index']] = torch.cat([self.label_passrate_list[extra_info['index']], tmp_label_question_accuracy_rate[i]], dim=0)
                                self.label_passrate_list[extra_info['index']] = torch.cat([self.label_passrate_list[extra_info['index']], tmp_label_question_accuracy_rate[i]], dim=0)
                                
                                 

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)


                        unlabel_solve_none = 0
                        unlabel_solve_all = 0
                        unlabel_solve_none_format = 0
                        unlabel_question_accuracy_rate = torch.zeros([unlabel_reward_tensor.shape[0],]).to(reward_tensor.device)
                        real_unlabel_question_accuracy_rate = torch.zeros([real_unlabel_reward_tensor.shape[0],]).to(reward_tensor.device)

                        for unlabel_uid in unlabel_unique_uids:
                            unlabel_uid_mask = unlabel_uids == unlabel_uid

                            unlabel_uid_rewards = unlabel_reward_tensor[unlabel_uid_mask].sum(-1)  # Sum rewards for each sequence
                            unlabel_accuracy_rate = unlabel_uid_rewards.sum(0) / unlabel_uid_rewards.shape[0]
                            unlabel_question_accuracy_rate[unlabel_uid_mask] = unlabel_accuracy_rate

                            real_unlabel_uid_rewards = real_unlabel_reward_tensor[unlabel_uid_mask].sum(-1)  # Sum rewards for each sequence
                            real_unlabel_accuracy_rate = real_unlabel_uid_rewards.sum(0) / real_unlabel_uid_rewards.shape[0]
                            real_unlabel_question_accuracy_rate[unlabel_uid_mask] = real_unlabel_accuracy_rate
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (unlabel_uid_rewards == fail_value).all():
                                unlabel_valid_mask[unlabel_uid_mask] = False
                                unlabel_solve_none += 1
                            elif (unlabel_uid_rewards == success_value).all():
                                unlabel_valid_mask[unlabel_uid_mask] = False
                                unlabel_solve_all += 1
                            elif (unlabel_uid_rewards == format_value).all():
                                unlabel_valid_mask[unlabel_uid_mask] = False
                                unlabel_solve_none_format += 1

                        unlabel_mean_accuracy_rate = torch.mean(unlabel_question_accuracy_rate, dim=0)
                        metrics['ul_batch/mean_accuracy_rate'] = unlabel_mean_accuracy_rate.item()
                        unlabel_batch.batch['accuracy_rate'] = unlabel_question_accuracy_rate


                        current_unlabel_passrate_list = {}
                        current_index_1 = []
                        tmp_unlabel_question_accuracy_rate = unlabel_question_accuracy_rate.unsqueeze(1)
                        for i, extra_info in enumerate(unlabel_batch.non_tensor_batch['extra_info']):
                            if extra_info['index'] in current_index_1:
                                continue
                            current_index_1.append(extra_info['index'])
                            if extra_info['index'] not in self.unlabel_passrate_list.keys():
                                self.unlabel_passrate_list[extra_info['index']] = tmp_unlabel_question_accuracy_rate[i]
                                current_unlabel_passrate_list[extra_info['index']] = tmp_unlabel_question_accuracy_rate[i]
                                if current_unlabel_passrate_list[extra_info['index']].shape[0] > self.max_length:
                                    self.max_length = current_unlabel_passrate_list[extra_info['index']].shape[0]
                            else:
                                current_unlabel_passrate_list[extra_info['index']] = torch.cat([self.unlabel_passrate_list[extra_info['index']], tmp_unlabel_question_accuracy_rate[i]], dim=0)
                                self.unlabel_passrate_list[extra_info['index']] = torch.cat([self.unlabel_passrate_list[extra_info['index']], tmp_unlabel_question_accuracy_rate[i]], dim=0)
                                if current_unlabel_passrate_list[extra_info['index']].shape[0] > self.max_length:
                                    self.max_length = current_unlabel_passrate_list[extra_info['index']].shape[0]


                        real_unlabel_mean_accuracy_rate = torch.mean(real_unlabel_question_accuracy_rate, dim=0)
                        metrics['ul_batch/real_mean_accuracy_rate'] = real_unlabel_mean_accuracy_rate.item()
                        unlabel_batch.batch['real_accuracy_rate'] = real_unlabel_question_accuracy_rate



                        current_real_unlabel_passrate_list = {}
                        current_index_2 = []
                        tmp_real_unlabel_question_accuracy_rate = real_unlabel_question_accuracy_rate.unsqueeze(1)
                        for i, extra_info in enumerate(unlabel_batch.non_tensor_batch['extra_info']):
                            if extra_info['index'] in current_index_2:
                                continue
                            current_index_2.append(extra_info['index'])
                            if extra_info['index'] not in self.real_unlabel_passrate_list.keys():
                                self.real_unlabel_passrate_list[extra_info['index']] = tmp_real_unlabel_question_accuracy_rate[i]
                                current_real_unlabel_passrate_list[extra_info['index']] = tmp_real_unlabel_question_accuracy_rate[i]
                            else:
                                current_real_unlabel_passrate_list[extra_info['index']] = torch.cat([self.real_unlabel_passrate_list[extra_info['index']], tmp_real_unlabel_question_accuracy_rate[i]], dim=0)
                                self.real_unlabel_passrate_list[extra_info['index']] = torch.cat([self.real_unlabel_passrate_list[extra_info['index']], tmp_real_unlabel_question_accuracy_rate[i]], dim=0)
                                         
                        
                       
                        diff_len = len(current_index_1)
                        candidate_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_1_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_2_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_3_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_4_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_5_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_6_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_7_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_8_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_0_9_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)
                        top_final_unlabel_mask = torch.zeros(8*diff_len, dtype=torch.bool)


                        metrics['ul_batch/track_max_length'] = self.max_length

                        if not self.config.actor_rollout_ref.use_trend_match:
                            top_final_unlabel_mask = torch.ones(8*diff_len, dtype=torch.bool)
                            unlabel_batch.batch['candidate_unlabel_mask'] = top_final_unlabel_mask
                            metrics['ul_batch/semi_mode'] = 0.0
                            # metrics['ul_batch/num_candidate'] = 0.0
                            true_count = top_final_unlabel_mask.sum().item()
                            metrics['ul_batch/num_candidate'] = true_count
                            metrics['ul_batch/ref_len'] = 0.0
                            metrics['ul_batch/ref_re_init'] = 0.0
                            metrics['ul_batch/use_trend_match'] = 0.0
                        else:
                        
                            if self.max_length >= self.config.actor_rollout_ref.start_semi_epoch:  

                                metrics['ul_batch/ref_re_init'] = 0.0
                               
                                if self.max_length != self.old_max_length:
                                    metrics['ul_batch/ref_re_init'] = 1.0
                                    self.old_max_length = self.max_length

                                self.copy_dict_with_validation(current_label_passrate_list, prefix="label", filter_by_max_length=False)
                                top_indices, top_scores, top_keys, \
                                top_0_1_scores, top_0_2_scores, top_0_3_scores, top_0_4_scores, top_0_5_scores, top_0_6_scores, top_0_7_scores, top_0_8_scores, top_0_9_scores, final_scores, \
                                top_0_1_keys, top_0_2_keys, top_0_3_keys, top_0_4_keys, top_0_5_keys, top_0_6_keys, top_0_7_keys, top_0_8_keys, top_0_9_keys, final_keys = \
                                self.find_top_related_cosine(query_dict = self.ref_passrate_list, database_dict = current_unlabel_passrate_list, top_k_ratio=0.1, aggregate_method='mean')
                                metrics['ul_batch/mean_final_candidate_scores'] = final_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_candidate_scores'] = top_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_1_scores'] = top_0_1_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_2_scores'] = top_0_2_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_3_scores'] = top_0_3_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_4_scores'] = top_0_4_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_5_scores'] = top_0_5_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_6_scores'] = top_0_6_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_7_scores'] = top_0_7_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_8_scores'] = top_0_8_scores.mean(dim=0).item()
                                metrics['ul_batch/mean_top_0_9_scores'] = top_0_9_scores.mean(dim=0).item()
                                
                                

                                for key, key_0_1, key_0_2, key_0_3, key_0_4, key_0_5, key_0_6, key_0_7, key_0_8, key_0_9, final_key in zip_longest(top_keys, top_0_1_keys, top_0_2_keys, top_0_3_keys, top_0_4_keys, top_0_5_keys, top_0_6_keys, top_0_7_keys, top_0_8_keys, top_0_9_keys, final_keys, fillvalue=None):
                                    # breakpoint()
                                    if key is not None:
                                        k = current_index_1.index(key)
                                        start_idx = 8 * k
                                        end_idx = 8 * (k + 1)
                                        
                                        if start_idx < 8*diff_len:
                                            end_idx = min(end_idx, 8*diff_len)
                                            candidate_unlabel_mask[start_idx:end_idx] = True

                                    if key_0_1 is not None:
                                        k_0_1 = current_index_1.index(key_0_1)
                                        start_idx_0_1 = 8 * k_0_1
                                        end_idx_0_1 = 8 * (k_0_1 + 1)
                                        
                                        if start_idx_0_1 < 8*diff_len:
                                            end_idx_0_1 = min(end_idx_0_1, 8*diff_len)
                                            top_0_1_unlabel_mask[start_idx_0_1:end_idx_0_1] = True
                                    
                                    if key_0_2 is not None:
                                        k_0_2 = current_index_1.index(key_0_2)
                                        start_idx_0_2 = 8 * k_0_2
                                        end_idx_0_2 = 8 * (k_0_2 + 1)
                                        
                                        if start_idx_0_2 < 8*diff_len:
                                            end_idx_0_2 = min(end_idx_0_2, 8*diff_len)
                                            top_0_2_unlabel_mask[start_idx_0_2:end_idx_0_2] = True

                                    if key_0_3 is not None:
                                        k_0_3 = current_index_1.index(key_0_3)
                                        start_idx_0_3 = 8 * k_0_3
                                        end_idx_0_3 = 8 * (k_0_3 + 1)
                                        
                                        if start_idx_0_3 < 8*diff_len:
                                            end_idx_0_3 = min(end_idx_0_3, 8*diff_len)
                                            top_0_3_unlabel_mask[start_idx_0_3:end_idx_0_3] = True  
                                    
                                    if key_0_4 is not None:
                                        k_0_4 = current_index_1.index(key_0_4)
                                        start_idx_0_4 = 8 * k_0_4
                                        end_idx_0_4 = 8 * (k_0_4 + 1)
                                        
                                        if start_idx_0_4 < 8*diff_len:
                                            end_idx_0_4 = min(end_idx_0_4, 8*diff_len)
                                            top_0_4_unlabel_mask[start_idx_0_4:end_idx_0_4] = True

                                    if key_0_5 is not None:
                                        k_0_5 = current_index_1.index(key_0_5)
                                        start_idx_0_5 = 8 * k_0_5
                                        end_idx_0_5 = 8 * (k_0_5 + 1)
                                        
                                        if start_idx_0_5 < 8*diff_len:
                                            end_idx_0_5 = min(end_idx_0_5, 8*diff_len)
                                            top_0_5_unlabel_mask[start_idx_0_5:end_idx_0_5] = True

                                    if key_0_6 is not None:
                                        k_0_6 = current_index_1.index(key_0_6)
                                        start_idx_0_6 = 8 * k_0_6
                                        end_idx_0_6 = 8 * (k_0_6 + 1)
                                        
                                        if start_idx_0_6 < 8*diff_len:
                                            end_idx_0_6 = min(end_idx_0_6, 8*diff_len)
                                            top_0_6_unlabel_mask[start_idx_0_6:end_idx_0_6] = True

                                    if key_0_7 is not None:
                                        k_0_7 = current_index_1.index(key_0_7)
                                        start_idx_0_7 = 8 * k_0_7
                                        end_idx_0_7 = 8 * (k_0_7 + 1)
                                        
                                        if start_idx_0_7 < 8*diff_len:
                                            end_idx_0_7 = min(end_idx_0_7, 8*diff_len)
                                            top_0_7_unlabel_mask[start_idx_0_7:end_idx_0_7] = True

                                    if key_0_8 is not None:
                                        k_0_8 = current_index_1.index(key_0_8)
                                        start_idx_0_8 = 8 * k_0_8
                                        end_idx_0_8 = 8 * (k_0_8 + 1)
                                        
                                        if start_idx_0_8 < 8*diff_len:
                                            end_idx_0_8 = min(end_idx_0_8, 8*diff_len)
                                            top_0_8_unlabel_mask[start_idx_0_8:end_idx_0_8] = True

                                    if key_0_9 is not None:
                                        k_0_9 = current_index_1.index(key_0_9)
                                        start_idx_0_9 = 8 * k_0_9
                                        end_idx_0_9 = 8 * (k_0_9 + 1)
                                        
                                        if start_idx_0_9 < 8*diff_len:
                                            end_idx_0_9 = min(end_idx_0_9, 8*diff_len)
                                            top_0_9_unlabel_mask[start_idx_0_9:end_idx_0_9] = True      

                                    if final_key is not None:
                                        final_k = current_index_1.index(final_key)
                                        start_idx_f = 8 * final_k
                                        end_idx_f = 8 * (final_k + 1)
                                        
                                        if start_idx_f < 8*diff_len:
                                            end_idx_f = min(end_idx_f, 8*diff_len)
                                            top_final_unlabel_mask[start_idx_f:end_idx_f] = True                                                                                                                                                                                  

                                unlabel_batch.batch['candidate_unlabel_mask'] = top_final_unlabel_mask
                                true_count = top_final_unlabel_mask.sum().item()
                                metrics['ul_batch/num_candidate'] = true_count
                                
                                mean_all_real_passrate = real_unlabel_question_accuracy_rate.mean(dim=0)
                                metrics['ul_batch/mean_all_real_passrate'] = mean_all_real_passrate.item()

                                mean_final_candidate_real_passrate = real_unlabel_question_accuracy_rate[top_final_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_final_candidate_real_passrate'] = mean_final_candidate_real_passrate.item()

                                mean_candidate_real_passrate = real_unlabel_question_accuracy_rate[candidate_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_candidate_real_passrate'] = mean_candidate_real_passrate.item()

                                mean_top_0_1_real_passrate = real_unlabel_question_accuracy_rate[top_0_1_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_1_real_passrate'] = mean_top_0_1_real_passrate.item()

                                mean_top_0_2_real_passrate = real_unlabel_question_accuracy_rate[top_0_2_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_2_real_passrate'] = mean_top_0_2_real_passrate.item()

                                mean_top_0_3_real_passrate = real_unlabel_question_accuracy_rate[top_0_3_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_3_real_passrate'] = mean_top_0_3_real_passrate.item()

                                mean_top_0_4_real_passrate = real_unlabel_question_accuracy_rate[top_0_4_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_4_real_passrate'] = mean_top_0_4_real_passrate.item()

                                mean_top_0_5_real_passrate = real_unlabel_question_accuracy_rate[top_0_5_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_5_real_passrate'] = mean_top_0_5_real_passrate.item()

                                mean_top_0_6_real_passrate = real_unlabel_question_accuracy_rate[top_0_6_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_6_real_passrate'] = mean_top_0_6_real_passrate.item()

                                mean_top_0_7_real_passrate = real_unlabel_question_accuracy_rate[top_0_7_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_7_real_passrate'] = mean_top_0_7_real_passrate.item()

                                mean_top_0_8_real_passrate = real_unlabel_question_accuracy_rate[top_0_8_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_8_real_passrate'] = mean_top_0_8_real_passrate.item()

                                mean_top_0_9_real_passrate = real_unlabel_question_accuracy_rate[top_0_9_unlabel_mask].mean(dim=0)
                                metrics['ul_batch/mean_top_0_9_real_passrate'] = mean_top_0_9_real_passrate.item()                                                                                                                                                                                                    

                                top_unlabel_dict = {key: current_unlabel_passrate_list[key] for key in top_keys if key in current_unlabel_passrate_list}
                                self.copy_dict_with_validation(top_unlabel_dict, prefix="unlabel", filter_by_max_length=False)
                                metrics['ul_batch/semi_mode'] = 1.0
                                metrics['ul_batch/ref_len'] = len(self.ref_passrate_list)
                                metrics['ul_batch/use_trend_match'] = 1.0

                            else:
                                unlabel_batch.batch['candidate_unlabel_mask'] = None
                                metrics['ul_batch/semi_mode'] = 0.0
                                metrics['ul_batch/num_candidate'] = 0.0
                                metrics['ul_batch/ref_len'] = 0.0
                                metrics['ul_batch/ref_re_init'] = 0.0
                                metrics['ul_batch/use_trend_match'] = 1.0


                        if self.config.trainer.skip_valid_mask:
                            unlabel_valid_mask[:] = True
                        # Log to metrics
                        metrics['ul_batch/solve_none'] = unlabel_solve_none
                        metrics['ul_batch/solve_none_format'] = unlabel_solve_none_format
                        metrics['ul_batch/solve_all'] = unlabel_solve_all

                        # add more metrics
                        metrics['ul_batch/solved'] = (unlabel_reward_tensor.sum(-1) == success_value).sum().item() / len(unlabel_uids)
                        metrics['ul_batch/failed'] = (unlabel_reward_tensor.sum(-1) == fail_value).sum().item() / len(unlabel_uids)
                        # add on-policy metrics
                        unlabel_prefix_mask = unlabel_batch.batch['prefix_mask']
                        unlabel_off_policy_mask = unlabel_prefix_mask.any(-1)
                        unlabel_on_policy_mask = ~unlabel_off_policy_mask
                        metrics['ul_batch/on_solved'] = (unlabel_reward_tensor[unlabel_on_policy_mask].sum(-1) == success_value).sum().item() / (unlabel_on_policy_mask.sum().item() + 1e-6)
                        metrics['ul_batch/off_solved'] = (unlabel_reward_tensor[unlabel_off_policy_mask].sum(-1) == success_value).sum().item() / (unlabel_off_policy_mask.sum().item() + 1e-6)

                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                            unlabel_old_log_prob = self.actor_rollout_wg.compute_log_prob(unlabel_batch)
                            unlabel_batch = unlabel_batch.union(unlabel_old_log_prob)


                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                                unlabel_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(unlabel_batch)
                                unlabel_batch = unlabel_batch.union(unlabel_ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            unlabel_batch, unlabel_kl_metrics = apply_kl_penalty(unlabel_batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                            metrics.update(unlabel_kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                            unlabel_batch.batch['token_level_rewards'] = unlabel_batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                        
                        unlabel_batch = compute_advantage(unlabel_batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape

                        unlabel_prefix_mask = unlabel_batch.batch['prefix_mask']
                        unlabel_advantages = unlabel_batch.batch['advantages']
                        assert unlabel_prefix_mask.shape == unlabel_advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages

                        unlabel_alpha_weight = unlabel_prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        unlabel_beta_weight = (~unlabel_prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        unlabel_prefix_weight = unlabel_alpha_weight + unlabel_beta_weight
                        unlabel_batch.batch['advantages'] = unlabel_prefix_weight * unlabel_advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0


                            unlabel_responses = unlabel_batch.batch['responses']
                            unlabel_responses_mask = unlabel_responses != self.tokenizer.pad_token_id
                            unlabel_response_length = unlabel_responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            unlabel_has_truncated = unlabel_response_length >= max_len
                            unlabel_no_eos = ~((unlabel_responses == self.tokenizer.eos_token_id).any(-1))
                            unlabel_truncated_mask = unlabel_has_truncated & unlabel_no_eos
                            unlabel_batch.batch['advantages'][unlabel_truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                            unlabel_batch.batch['advantages'][unlabel_prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    self._balance_batch(unlabel_batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    unlabel_batch.meta_info['global_token_num'] = torch.sum(unlabel_batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                            unlabel_critic_output = self.critic_wg.update_critic(unlabel_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        unlabel_critic_output_metrics = reduce_metrics(unlabel_critic_output.meta_info['metrics'])
                        metrics.update(unlabel_critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch.meta_info['is_labeled'] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            unlabel_batch.meta_info['is_labeled'] = False
                            unlabel_batch.meta_info['use_ul_loss'] = self.config.actor_rollout_ref.use_unlabel_loss
                            unlabel_actor_output = self.actor_rollout_wg.update_actor(unlabel_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        unlabel_actor_output_metrics = reduce_metrics(unlabel_actor_output.meta_info['metrics'])
                        metrics.update(unlabel_actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                metrics.update(compute_data_metrics_ours(batch=unlabel_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=unlabel_batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
        
def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_on_example_ratio = off_policy_mask.sum().item() / on_policy_mask.sum().item()

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]


    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        # 'on_off_metrics/on_prompt_score':
        #     torch.mean(on_prompt_score).detach().item(),
        # 'on_off_metrics/off_prompt_score':
        #     torch.mean(off_prompt_score).detach().item(),
        'on_off_metrics/off_on_example_ratio':
            off_on_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics