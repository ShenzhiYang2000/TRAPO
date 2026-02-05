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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from deepscaler.rewards.math_reward import deepscaler_reward_fn, THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from typing import List, Union
from verl.mix_src.reward_with_format import deepscaler_reward_fn_impl1
from verl.mix_src.math_verify_reward import reward_fn_math_verify, reward_fn_math_verify_no_think
from math_verify import parse, verify
import random
from collections import Counter
def deepscaler_reward_fn_nothink(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    solution_str = f"{THOUGHT_DELIMITER_START}\n{THOUGHT_DELIMITER_END}\n{solution_str}"
    return deepscaler_reward_fn(solution_str, ground_truth, enable_llm)

def _select_rm_score_fn(data_source, reward_impl_version):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        if reward_impl_version == 0:
            return deepscaler_reward_fn
        elif reward_impl_version == 1:
            return deepscaler_reward_fn_impl1
        elif reward_impl_version == 2:
            return deepscaler_reward_fn_nothink
        elif reward_impl_version == 3:
            return reward_fn_math_verify
        elif reward_impl_version == 4:
            return reward_fn_math_verify_no_think
        else:
            raise NotImplementedError

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_impl_version = reward_impl_version

    def __call__(self, data: DataProto, is_labeled=True):
        if is_labeled:
            """We will expand this function gradually based on the available datasets"""
            # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
            if 'rm_scores' in data.batch.keys():
                return data.batch['rm_scores']

            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

            already_print_data_sources = {}

            from concurrent.futures import ThreadPoolExecutor
            from typing import Dict, Any
            #import threading
            # Thread-safe dict for tracking printed data sources
            # print_lock = threading.Lock()
            
            def process_item(args):
                i, data_item, already_print_data_sources = args
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses'] 
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences = valid_response_ids
                sequences_str = self.tokenizer.decode(sequences)
                # if not "no_think" in self.reward_impl_version:
                from deepscaler.globals import THOUGHT_DELIMITER_START
                # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
                if self.reward_impl_version != 4:
                    sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str
                # else:
                #     breakpoint()

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                # select rm_score
                data_source = data_item.non_tensor_batch['data_source']
                compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
                
                # with print_lock:
                #     if data_source not in already_print_data_sources:
                #         already_print_data_sources[data_source] = 0

                #     if already_print_data_sources[data_source] < self.num_examine:
                #         already_print_data_sources[data_source] += 1
                #         print(sequences_str)      
                return i, score, valid_response_length

            if self.reward_impl_version in {3, 4}:
                args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                results = list(process_item(args[i]) for i in range(len(args)))
            else:
                # Process items in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=96) as executor:
                    args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                    results = list(executor.map(process_item, args))

            # Fill reward tensor with results
            for i, score, valid_response_length in results:
                reward_tensor[i, valid_response_length - 1] = score

            return reward_tensor

        else:
            """We will expand this function gradually based on the available datasets"""
            # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
            if 'rm_scores' in data.batch.keys():
                return data.batch['rm_scores']

            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

            already_print_data_sources = {}

            from concurrent.futures import ThreadPoolExecutor
            from typing import Dict, Any
            #import threading
            # Thread-safe dict for tracking printed data sources
            # print_lock = threading.Lock()

            def process_list_concise(new_list, group_size=8):
                result = []
                
                for i in range(0, len(new_list), group_size):
                    group = new_list[i:i + group_size]
                    counter = Counter(group)
                    
                 
                    # replace_value = (counter.most_common(1)[0][0] 
                    #                 if counter.most_common(1)[0][1] > 1 
                    #                 else random.choice(group))

                          
                    # replace_value = (counter.most_common(1)[0][0] 
                    #                 if counter.most_common(1)[0][1] > 1 
                    #                 else 'MissMiss')

                    common = counter.most_common()
                    replace_value = (common[0][0] 
                    if common and common[0][1] > 1 and 
                        (len(common) == 1 or common[1][1] < common[0][1]) 
                    else 'MissMiss')
                        
                    result.extend([replace_value] * len(group))
                
                return result

            def extract_vote_answer(args):
                i, data_item, already_print_data_sources = args
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses'] 
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences = valid_response_ids
                sequences_str = self.tokenizer.decode(sequences)
                # if not "no_think" in self.reward_impl_version:
                from deepscaler.globals import THOUGHT_DELIMITER_START
                # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
                if self.reward_impl_version != 4:
                    sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str
                # else:
                #     breakpoint()

                return sequences_str


            
            def process_item_unlabel(args, pesudo_label):
                i, data_item, already_print_data_sources = args
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses'] 
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences = valid_response_ids
                sequences_str = self.tokenizer.decode(sequences)
                # if not "no_think" in self.reward_impl_version:
                from deepscaler.globals import THOUGHT_DELIMITER_START
                # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
                if self.reward_impl_version != 4:
                    sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str

                real_ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                data_source = data_item.non_tensor_batch['data_source']
                compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
                score = compute_score_fn(solution_str=sequences_str, ground_truth=pesudo_label)
                   
                return i, score, valid_response_length 

            if self.reward_impl_version in {3, 4}:
                args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                generate_responses = list(extract_vote_answer(args[i]) for i in range(len(args)))
                predict_answers = list(map(parse, generate_responses))
                verified_predict_answers = [item[1] if len(item)>=2 else 'None' for item in predict_answers]
                pesudo_ground_truth = process_list_concise(verified_predict_answers, group_size=8)

                
                results = list(process_item_unlabel(args[i], pesudo_label = pesudo_ground_truth[i]) for i in range(len(args)))
            else:
                # Process items in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=96) as executor:
                    args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                    results = list(executor.map(process_item_unlabel, args))

            # Fill reward tensor with results
            for i, score, valid_response_length in results:
                reward_tensor[i, valid_response_length - 1] = score

            return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='mix_ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        raise NotImplementedError('megatron is not supported')
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from .mix_fsdp_worker import MIXActorRolloutRefWorker

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    if config.actor_rollout_ref.ref.use_ref:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(MIXActorRolloutRefWorker)
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
    else:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, reward_impl_version=config.data.reward_impl_version)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, reward_impl_version=config.data.reward_impl_version)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    from .mix_trainer import MIXRayPPOTrainer
    if not config.trainer.acc_rebatch:
        trainer = MIXRayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
    else:
        from .mix_trainer_acc_rebatch import MIXRayPPOTrainerAccRebatch
        trainer = MIXRayPPOTrainerAccRebatch(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit_semi_supervised()


if __name__ == '__main__':
    main()
