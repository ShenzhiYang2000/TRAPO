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
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import (
    # compute_data_metrics,
    compute_data_metrics_semi,
    # compute_throughout_metrics,
    # compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from math_verify import parse, verify

import copy
import torch.nn.functional as F
# import math
from scipy.stats import linregress
import random
from collections import Counter
import ray
from verl.experimental.dataset.sampler import AbstractCurriculumSampler




class RaySemiTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self, *args, **kwargs):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        super().__init__(*args, **kwargs)

        self.real_gt = {}
        self.majority_answer = {}
        self.majority_answer_list = defaultdict(list)
        self.sample_is_right = {}

        self.label_passrate_list = {}
        self.unlabel_passrate_list = {}
        self.real_unlabel_passrate_list = {}
        self.ref_passrate_list = {}

        self.best_score = -100.0

        self.resume_from_epoch = 0






    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            # if "reward_extra_info" in result:
            #     for key, lst in result["reward_extra_info"].items():
            #         reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            # breakpoint()
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        core_metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

                    if metric_sec == "val-core" and "mean@" in metric_name:
                        pfx = f"{data_source}/{metric_name}"
                        core_metric_dict[pfx] = metric_val

        if core_metric_dict:
            avg_score = sum(core_metric_dict.values()) / len(core_metric_dict)
        else:
            avg_score = 0.0

        output_dict = {**core_metric_dict, 'average': avg_score}
        
        if avg_score > self.best_score:
            self.best_score = avg_score
            # if self.config.trainer.get("save_best_only", False):
            self._save_checkpoint()

                # Write validation results to file
        log_dir = self.config.trainer.get("default_local_dir", None)
        if log_dir:
            log_path = os.path.join(log_dir, f"valid/global_step_{self.global_steps}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                for key, value in output_dict.items():
                    f.write(f"{key}: {value}\n")


        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict




    def _save_trainer_state(self, checkpoint_dir: str):
        """
        Save custom trainer state (e.g., passrate, consensus, trust flags) to checkpoint_dir/trainer_state.pt.
        """
        import torch
        import os

        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        
        trainer_state = {
            # Core custom metrics and buffers
            'real_gt': self.real_gt,
            'majority_answer': self.majority_answer,
            'sample_is_right': self.sample_is_right,

            'label_passrate_list': self.label_passrate_list,
            'unlabel_passrate_list': self.unlabel_passrate_list,
            'real_unlabel_passrate_list': self.real_unlabel_passrate_list,
            'ref_passrate_list': self.ref_passrate_list,

            'best_score': self.best_score,

            # Optional: add version for future compatibility
            'version': '1.0',
        }

        torch.save(trainer_state, trainer_state_path)
        print(f"[Checkpoint] Custom trainer state saved to {trainer_state_path}")


    def _load_trainer_state(self, checkpoint_dir: str):
        """
        Load custom trainer state from checkpoint_dir/trainer_state.pt if exists.
        Returns True if loaded successfully, False otherwise.
        """
        import torch
        import os

        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            print(f"[Checkpoint] No custom trainer state found at {trainer_state_path}")
            return False

        trainer_state = torch.load(trainer_state_path, map_location='cpu', weights_only=False)

        # Restore attributes with safe defaults
        self.real_gt = trainer_state.get('real_gt', {})
        self.majority_answer = trainer_state.get('majority_answer', {})
        self.sample_is_right = trainer_state.get('sample_is_right', {})

        self.label_passrate_list = trainer_state.get('label_passrate_list', {})
        self.unlabel_passrate_list = trainer_state.get('unlabel_passrate_list', {})
        self.real_unlabel_passrate_list = trainer_state.get('real_unlabel_passrate_list', {})
        self.ref_passrate_list = trainer_state.get('ref_passrate_list', {})

        # compute min lengths
        label_min_len = min(t.size(0) for t in self.label_passrate_list.values())
        unlabel_min_len = min(t.size(0) for t in self.unlabel_passrate_list.values())

        # choose k
        k = min(label_min_len, unlabel_min_len)

        self.resume_from_epoch = k

        # in-place slice values
        for key in self.label_passrate_list:
            self.label_passrate_list[key] = self.label_passrate_list[key][:k]

        for key in self.unlabel_passrate_list:
            self.unlabel_passrate_list[key] = self.unlabel_passrate_list[key][:k]

        for key in self.real_unlabel_passrate_list:
            self.real_unlabel_passrate_list[key] = self.real_unlabel_passrate_list[key][:k]

        for key in self.ref_passrate_list:
            self.ref_passrate_list[key] = self.ref_passrate_list[key][:k]


        self.best_score = trainer_state.get('best_score', 0.0)

        # breakpoint()

        version = trainer_state.get('version', 'unknown')
        print(f"[Checkpoint] Custom trainer state loaded from {trainer_state_path} (version: {version})")
        return True

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # >>>>>>>>>>>> ADD THIS LINE <<<<<<<<<<<<
        self._save_trainer_state(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # >>>>>>>>>>>> ADD THIS LINE <<<<<<<<<<<<
        self._load_trainer_state(global_step_folder)





    def _get_global_mean_passrate(self, data: DataProto, majority_or_real: str):
        if majority_or_real == 'majority':
            global_majority_passrate = torch.tensor([data_item.non_tensor_batch["extra_info"]['majority_passrate'] for data_item in data]).mean()
            metrics = {"global_mean_majority_passrate": global_majority_passrate.item()}
            # breakpoint()
            return metrics
        elif majority_or_real == 'real':
            global_real_passrate = torch.tensor([data_item.non_tensor_batch["extra_info"]['real_passrate'] for data_item in data]).mean()
            # breakpoint()
            metrics = {"global_mean_real_passrate": global_real_passrate.item()}
            return metrics
        else:
            raise NotImplementedError("please input 'majority' or 'real'")        

    def _get_majority_accuracy(self, data: DataProto, labeled_or_unlabeled_or_global: str):
        if labeled_or_unlabeled_or_global == 'unlabeled':
            unlabeled_majority_is_right_list = []
            unfiltered_unlabeled_cnt = 0
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                assert extra_info and extra_info["labeled"] in (True, False)
                if extra_info["labeled"] is False:
                    unfiltered_unlabeled_cnt += 1
                    unlabeled_majority_is_right_list.append(data_item.non_tensor_batch["extra_info"]['majority_is_right'])
            if len(unlabeled_majority_is_right_list) == 0: 
                unlabeled_majority_accuracy = None
            else:
                unlabeled_majority_accuracy = torch.tensor(unlabeled_majority_is_right_list).float().mean()
            metrics = {"unlabeled_majority_accuracy": unlabeled_majority_accuracy.item(), "unfiltered_unlabeled_cnt":unfiltered_unlabeled_cnt}
            return metrics
        elif labeled_or_unlabeled_or_global == 'labeled':
            labeled_majority_is_right_list = []
            unfiltered_labeled_cnt = 0
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                assert extra_info and extra_info["labeled"] in (True, False)
                if extra_info["labeled"] is True:
                    unfiltered_labeled_cnt += 1
                    labeled_majority_is_right_list.append(data_item.non_tensor_batch["extra_info"]['majority_is_right'])
            if len(labeled_majority_is_right_list) == 0:
                labeled_majority_accuracy = None
            else:
                labeled_majority_accuracy = torch.tensor(labeled_majority_is_right_list).float().mean()
            metrics = {"labeled_majority_accuracy": labeled_majority_accuracy.item(), "unfiltered_labeled_cnt":unfiltered_labeled_cnt}
            return metrics
        elif labeled_or_unlabeled_or_global == 'global':
            global_majority_is_right_list = []
            unfiltered_all_cnt = 0
            for i in range(len(data)):
                unfiltered_all_cnt += 1
                data_item = data[i]  # DataProtoItem
                global_majority_is_right_list.append(data_item.non_tensor_batch["extra_info"]['majority_is_right'])
            global_majority_accuracy = torch.tensor(global_majority_is_right_list).float().mean()
            metrics = {"global_majority_accuracy": global_majority_accuracy.item(),"unfiltered_all_cnt":unfiltered_all_cnt}
            return metrics
        else:
            raise NotImplementedError("please input 'labeled' or 'unlabeled' or 'global'")      



    def _generate_pseudo_labels_and_update_passrate(self, data: DataProto, epoch):
        """Get passrate of each samples"""
        uid2idx = defaultdict(list)

        # map uid to indices for unlabeled data
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            data_item.non_tensor_batch["extra_info"]['majority_passrate'] = None
            data_item.non_tensor_batch["extra_info"]['real_passrate'] = None

            uid = data_item.non_tensor_batch.get("uid")
            uid2idx[uid].append(i)

        # majority voting to generate pseudo-labels
        for uid, indices in uid2idx.items():
            extra_info = data[indices[0]].non_tensor_batch.get("extra_info", {})
            prompt_index = extra_info['index'] 
            is_labeled = extra_info["labeled"]
            responses = []

            for idx in indices:
                data_item = data[idx]
                
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                responses.append(response_str)

            predict_answers = list(map(parse, responses))
            # rollout_n = self.config.actor_rollout_ref.rollout.n

            # # check exp_tower to avoid timeout
            # def has_exp_tower(s):
            #     s = s.replace(" ", "").replace("^", "**")
            #     return bool(re.search(r"\*\*\s*\(.*\*\*", s)) or s.count("**") >= 2

            # calculate the frequency of all answers
            freq_list = []
            for answer in predict_answers:
                found = False
                for k, (exist_answer, count) in enumerate(freq_list):
                    try:
                        # if exist_answer[1] != answer[1] and (has_exp_tower(exist_answer[1]) or has_exp_tower(answer[1])):
                        #     found = False
                        #     break
                        if exist_answer[1] == answer[1] or verify(exist_answer, answer):
                            freq_list[k] = (exist_answer, count + 1)
                            found = True
                            break
                    except Exception:
                        pass
                if not found:
                    freq_list.append((answer, 1))

            # find the majority answer as pseudo ground-truth
            counts = [count for _, count in freq_list]
            max_count = max(counts)
            count_freq = Counter(counts)


            if is_labeled:
                real_reward_tensor, _ = compute_reward(data.select_idxs(indices), self.reward_fn)
                real_scores = real_reward_tensor.sum(-1).cpu().tolist()
                real_passrate = torch.tensor(sum(real_scores) / len(real_scores)).unsqueeze(0)
                ground_truth = data[indices[0]].non_tensor_batch["reward_model"]['ground_truth']
                # majority_answer = next(rep[1] for rep, count in freq_list if count == max_count)
                try:
                    majority_answer = next(rep[1] for rep, count in freq_list if count == max_count)
                except Exception:
                    majority_answer = '__NoMajorityAnswer__'
                self.majority_answer[prompt_index] = majority_answer
                self.majority_answer_list[prompt_index].append(majority_answer)
                if prompt_index not in self.label_passrate_list:
                    self.label_passrate_list[prompt_index] = real_passrate
                else:
                    self.label_passrate_list[prompt_index] = torch.cat([
                        self.label_passrate_list[prompt_index], real_passrate
                    ], dim=0)   
                ma_is_right = verify(ground_truth, majority_answer)
                self.sample_is_right[prompt_index] = ma_is_right
                for i, idx in enumerate(indices):
                    data[idx].non_tensor_batch["extra_info"]['majority_passrate'] = real_passrate.item() # [0.0, 1.0] # watch out!
                    data[idx].non_tensor_batch["extra_info"]['real_passrate'] = real_passrate.item() # [0.0, 1.0]
                    # if majority_answer == ground_truth:
                    if ma_is_right:
                        data[idx].non_tensor_batch["extra_info"]['majority_is_right'] = True
                    else:
                        data[idx].non_tensor_batch["extra_info"]['majority_is_right'] = False
            else:
                for idx in indices:
                    if prompt_index in self.real_gt:
                        data[idx].non_tensor_batch["reward_model"]["ground_truth"] = self.real_gt[prompt_index]
                real_reward_tensor, _ = compute_reward(data.select_idxs(indices), self.reward_fn)
                real_scores = real_reward_tensor.sum(-1).cpu().tolist()
                real_passrate = torch.tensor(sum(real_scores) / len(real_scores)).unsqueeze(0)
                if prompt_index not in self.real_unlabel_passrate_list:
                    self.real_unlabel_passrate_list[prompt_index] = real_passrate
                else:
                    self.real_unlabel_passrate_list[prompt_index] = torch.cat([
                        self.real_unlabel_passrate_list[prompt_index], real_passrate
                    ], dim=0) 

                ground_truth = data[indices[0]].non_tensor_batch["reward_model"]['ground_truth'] 

                if count_freq[max_count] == 1:
                    majority_answer = next(rep[1] for rep, count in freq_list if count == max_count)
                else:
                    majority_answer = '__NoMajorityAnswer__'
                    
                self.majority_answer[prompt_index] = majority_answer

                self.majority_answer_list[prompt_index].append(majority_answer)
                for idx in indices:
                    if prompt_index not in self.real_gt and 'ground_truth' in data[idx].non_tensor_batch["reward_model"]:
                        self.real_gt[prompt_index] = data[idx].non_tensor_batch["reward_model"]["ground_truth"]
                    data[idx].non_tensor_batch["reward_model"]["ground_truth"] = majority_answer
                majority_reward_tensor, _ = compute_reward(data.select_idxs(indices), self.reward_fn)
                majority_scores = majority_reward_tensor.sum(-1).cpu().tolist()
                majority_passrate = torch.tensor(sum(majority_scores) / len(majority_scores)).unsqueeze(0)

                if prompt_index not in self.unlabel_passrate_list:
                    self.unlabel_passrate_list[prompt_index] = majority_passrate
                else:
                    self.unlabel_passrate_list[prompt_index] = torch.cat([
                        self.unlabel_passrate_list[prompt_index], majority_passrate
                    ], dim=0)  

                ma_is_right = verify(ground_truth, majority_answer)
                self.sample_is_right[prompt_index] = ma_is_right
                for idx in indices:
                    data[idx].non_tensor_batch["extra_info"]['majority_passrate'] = majority_passrate.item() # [0.0, 1.0] # watch out!
                    data[idx].non_tensor_batch["extra_info"]['real_passrate'] = real_passrate.item() # [0.0, 1.0] # watch out!
 
                    if ma_is_right:
                        data[idx].non_tensor_batch["extra_info"]['majority_is_right'] = True
                    else:
                        data[idx].non_tensor_batch["extra_info"]['majority_is_right'] = False

        return data




    def copy_dict_with_validation(self, original_dict, prefix="label", filter_by_max_length=False):
        """
        将原始字典的键值对复制到新字典中，添加前缀，并进行冲突检查
        
        Args:
            original_dict: 原始字典
            new_dict: 目标字典（已初始化的空字典）
            prefix: 要添加的前缀
        
        Returns:
            更新后的新字典
        """

        for key, value in original_dict.items():
            new_key = f"{prefix}_{key}"
            
            if new_key.startswith('unlabel') and new_key in self.ref_passrate_list:
                self.ref_passrate_list = {k: v for k, v in self.ref_passrate_list.items() if not k.startswith('unlabel')}
                self.ref_passrate_list[new_key] = value
            else:
                self.ref_passrate_list[new_key] = value



    def _get_slope(self, vector):
        # 假设你的向量是一个一维 PyTorch Tensor
        # 转换为 NumPy 数组（确保在 CPU 上）

        if vector.numel() < 2:
            return 0.0 

        y = vector.detach().cpu().numpy()
        x = np.arange(len(y))

        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return slope

    def filter_dict_by_topk(self, aggregate_scores: torch.Tensor, uid2idx: dict, top_k_ratio: float):
        """
        根据 aggregate_scores（shape: [num_queries]）和 uid2idx，
        返回被选中的 uid 集合（set），用于后续过滤。
        
        前提：aggregate_scores 的顺序与 list(uid2idx.keys()) 严格一致。
        """
        uids = list(uid2idx.keys())
        n = len(uids)
        assert aggregate_scores.numel() == n, \
            f"aggregate_scores length ({aggregate_scores.numel()}) != number of uids ({n})"

        # 至少选 1 个
        k = max(1, int(n * top_k_ratio))

        # 获取 top-k 的索引（在 uids 列表中的位置）
        _, topk_indices = torch.topk(aggregate_scores, k, largest=True, sorted=False)

        # 构建 selected uid set
        selected_uids = {uids[i] for i in topk_indices.tolist()}
        return selected_uids

    def filter_dict_by_threshold(self, aggregate_scores: torch.Tensor, uid2idx: dict, threshold: float) -> set:
        """
        根据 aggregate_scores 和 uid2idx，返回相似度 >= threshold 的 uid 集合。
        
        Args:
            aggregate_scores: (N,) tensor, scores for each uid (in order of uid2idx.keys())
            uid2idx: dict mapping uid -> indices
            threshold: similarity threshold
        
        Returns:
            set of selected uids
        """
        uids = list(uid2idx.keys())
        assert len(uids) == aggregate_scores.numel(), "Length mismatch"

        # 转为 CPU numpy 或直接用 torch 比较
        mask = aggregate_scores >= threshold  # (N,) bool tensor
        selected_uids = {uids[i] for i, is_selected in enumerate(mask.tolist()) if is_selected}
        return selected_uids

    def _compute_trajectory_cosine(self, data, top_k_ratio=0.1, threshold=0.4, aggregate_method='mean', filter = True):
        """Generate pseudo-labels for unlabeled samples and filter out those lacking consensus"""
        uid2idx = defaultdict(list)
        keep_indices = []

        # map uid to indices for unlabeled data
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            assert extra_info and extra_info["labeled"] in (True, False)
            # breakpoint()

            if extra_info["labeled"] is False:
                uid = data_item.non_tensor_batch.get("uid")
                uid2idx[uid].append(i)
            else:
                keep_indices.append(i)

        database_tensors = list(self.ref_passrate_list.values())
        ref_cnt = len(database_tensors)
        # 假设 database_tensors 是 [tensor1, tensor2, ...]，每个 shape=(L_i, ...)
        if not database_tensors:
            raise ValueError('database_tensors is empty')
        else:
            min_len = min(t.size(0) for t in database_tensors)
            max_len = max(t.size(0) for t in database_tensors)
            truncated_tensors = [t[:min_len] for t in database_tensors]

        # database_keys = list(self.ref_passrate_list.keys())
        database = torch.stack(truncated_tensors, dim=0)

        # uid2slope = {}
        uid2sim = {}
        uid2query = {}
        for uid, indices in uid2idx.items():
            extra_info = data[indices[0]].non_tensor_batch.get("extra_info", {})
            prompt_index = extra_info['index'] 
            query_tensor = self.unlabel_passrate_list[prompt_index][:min_len]
            if query_tensor.size(0) < min_len:
                raise ValueError("query_tensor length is less than min_len")
            uid2query[uid] = query_tensor

        query_tensors = list(uid2query.values())
        query = torch.stack(query_tensors, dim=0)

        # 检查形状
        if query.dim() != 2 or database.dim() != 2:
            raise ValueError("the tensor must be 2d (samples, features)")
        
        if query.shape[1] != database.shape[1]:
            raise ValueError(f"feature dimension is not match: query={query.shape[1]}, database={database.shape[1]}")

        query_norm = (query - query.mean(dim=1, keepdim=True)) / (query.std(dim=1, keepdim=True) + 1e-8)
        db_norm = (database - database.mean(dim=1, keepdim=True)) / (database.std(dim=1, keepdim=True) + 1e-8)

        db_slope = self._get_slope(db_norm.mean(dim=0))

        cosine_sim = F.cosine_similarity(db_norm.unsqueeze(1), query_norm.unsqueeze(0), dim=2)
        
        if aggregate_method == 'mean':
            aggregate_scores = cosine_sim.mean(dim=0)  
        elif aggregate_method == 'max':
            aggregate_scores = cosine_sim.max(dim=0).values  
        else:
            raise ValueError("aggregate_method must be 'mean' or 'max'")

        uid2sim = dict(zip(uid2idx.keys(), aggregate_scores.tolist()))

        dict_filter_by_topk = self.filter_dict_by_topk(aggregate_scores, uid2idx, top_k_ratio)

        dict_filter_by_threshold = self.filter_dict_by_threshold(aggregate_scores, uid2idx, threshold)

        final_dict = dict_filter_by_topk | dict_filter_by_threshold  # set union

        unlabeled_cnt = 0
        selected_unlabeled_cnt = 0
        selected_right_unlabeled_cnt = 0
        left_unlabeled_cnt = 0
        left_right_unlabeled_cnt = 0

        final_prompt_idx = []

        selected_sim = 0.0
        selected_real_passrate = 0.0
        left_sim = 0.0
        left_real_passrate = 0.0

        selected_slope = 0.0
        left_slope = 0.0


        for uid, indices in uid2idx.items():
            extra_info = data[indices[0]].non_tensor_batch.get("extra_info", {})
            prompt_index = extra_info['index'] 
            unlabeled_cnt += 1
            if uid in final_dict:
                selected_sim += uid2sim[uid]
                selected_real_passrate += data[indices[0]].non_tensor_batch["extra_info"]['real_passrate']
                selected_unlabeled_cnt += 1
                selected_slope += self._get_slope(self.unlabel_passrate_list[prompt_index][:min_len])
                final_prompt_idx.append(prompt_index)
                # if data[indices[0]].non_tensor_batch["extra_info"]['majority_is_right']:
                if self.sample_is_right[prompt_index]:
                    selected_right_unlabeled_cnt += 1
                for idx in indices:
                    keep_indices.append(idx)
            else:
                left_sim += uid2sim[uid]
                left_real_passrate += data[indices[0]].non_tensor_batch["extra_info"]['real_passrate']
                left_unlabeled_cnt += 1
                left_slope += self._get_slope(self.unlabel_passrate_list[prompt_index][:min_len])
                # if data[indices[0]].non_tensor_batch["extra_info"]['majority_is_right']:
                if self.sample_is_right[prompt_index]:
                    left_right_unlabeled_cnt += 1

        if selected_unlabeled_cnt == 0:
            selected_sim = 0.0
            selected_real_passrate = 0.0
            selected_slope = 0.0
        else:
            selected_sim /= selected_unlabeled_cnt
            selected_real_passrate /= selected_unlabeled_cnt
            selected_slope /= selected_unlabeled_cnt

        if left_unlabeled_cnt == 0:
            left_sim = 0.0
            left_real_passrate = 0.0
            left_slope = 0.0
        else:
            left_sim /= left_unlabeled_cnt
            left_real_passrate /= left_unlabeled_cnt
            left_slope /= left_unlabeled_cnt
        # breakpoint()

        metrics = {"ref_cnt":ref_cnt, "ref_min_len":min_len, "ref_max_len":max_len, "unlabeled_cnt": unlabeled_cnt, "selected_unlabeled_cnt": selected_unlabeled_cnt, "selected_right_unlabeled_cnt": selected_right_unlabeled_cnt, "left_unlabeled_cnt": left_unlabeled_cnt, "left_right_unlabeled_cnt":left_right_unlabeled_cnt, \
            "db_slope":db_slope, "selected_slope":selected_slope, "left_slope":left_slope, "selected_sim":selected_sim, "left_sim":left_sim, "selected_real_passrate":selected_real_passrate, "left_real_passrate":left_real_passrate}
        
        if filter:
            data = data.select_idxs(keep_indices)

        return data, metrics, final_prompt_idx



    def _split_labeled_samples(self, data):
        """Generate pseudo-labels for unlabeled samples and filter out those lacking consensus"""
        # uid2idx = defaultdict(list)
        keep_indices = []

        # map uid to indices for unlabeled data
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            assert extra_info and extra_info["labeled"] in (True, False)
            # breakpoint()
            if extra_info["labeled"] is True:
                keep_indices.append(i)

        data = data.select_idxs(keep_indices)
        
        return data



    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            log_file=self.config.trainer.output_log_path
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # for epoch in range(self.config.trainer.total_epochs):
        for epoch in range(self.resume_from_epoch, self.config.trainer.total_epochs, 1):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)



                    # generate pseudo labels for unlabeled data
                    if self.config.trainer.train_mode != "supervised":
                    # if False:
                        batch = self._generate_pseudo_labels_and_update_passrate(batch, epoch)

                        metrics_global_mean_majority_passrate = self._get_global_mean_passrate(batch, majority_or_real = 'majority')
                        metrics_global_mean_real_passrate = self._get_global_mean_passrate(batch, majority_or_real = 'real')

                        metrics.update(metrics_global_mean_majority_passrate)
                        metrics.update(metrics_global_mean_real_passrate)

                        metrics_global_majority_accuracy = self._get_majority_accuracy(batch, labeled_or_unlabeled_or_global = 'global')
                        metrics_labeled_majority_accuracy = self._get_majority_accuracy(batch, labeled_or_unlabeled_or_global = 'labeled')
                        metrics_unlabeled_majority_accuracy = self._get_majority_accuracy(batch, labeled_or_unlabeled_or_global = 'unlabeled')
                        
                        metrics.update(metrics_global_majority_accuracy)
                        metrics.update(metrics_labeled_majority_accuracy)
                        metrics.update(metrics_unlabeled_majority_accuracy)

                        if epoch >= self.config.trainer.warm_up:
                            self.copy_dict_with_validation(self.label_passrate_list, prefix="label", filter_by_max_length=False)
                            # if epoch == 4:
                            #     breakpoint()
                            batch, metrics_filter_majority_accuracy, selected_unlabeled_samples = self._compute_trajectory_cosine(batch, top_k_ratio=self.config.trainer.topk, threshold=self.config.trainer.thres, aggregate_method='mean', filter=True)
                            if self.config.trainer.update_db:
                                update_unlabeled_samples = {}
                                for k in selected_unlabeled_samples:
                                    update_unlabeled_samples[k] = copy.deepcopy(self.unlabel_passrate_list[k])
                                self.copy_dict_with_validation(update_unlabeled_samples, prefix="unlabel", filter_by_max_length=False)
                            metrics.update(metrics_filter_majority_accuracy)
                        else:
                            # self.copy_dict_with_validation(self.label_passrate_list, prefix="label", filter_by_max_length=False)
                            # batch, metrics_filter_majority_accuracy, selected_unlabeled_samples = self._compute_trajectory_cosine(batch, top_k_ratio=0.1, threshold=0.4, aggregate_method='mean', filter=False)
                            batch = self._split_labeled_samples(batch)
                            # if self.config.trainer.update_db:
                            #     update_unlabeled_samples = {}
                            #     for k in selected_unlabeled_samples:
                            #         update_unlabeled_samples[k] = copy.deepcopy(self.unlabel_passrate_list[k])
                            #     self.copy_dict_with_validation(update_unlabeled_samples, prefix="unlabel", filter_by_max_length=False)
                            # metrics.update(metrics_filter_majority_accuracy)


                        label_pr_list = list(self.label_passrate_list.values())
                        label_min_len = min(t.size(0) for t in label_pr_list)
                        label_max_len = max(t.size(0) for t in label_pr_list)
                        # assert label_max_len - label_min_len <= 1, "label_max_len - label_min_len > 1"
                        unlabel_pr_list = list(self.unlabel_passrate_list.values())
                        unlabel_min_len = min(t.size(0) for t in unlabel_pr_list)
                        unlabel_max_len = max(t.size(0) for t in unlabel_pr_list)
                        # assert unlabel_max_len - unlabel_min_len <= 1, "unlabel_max_len - unlabel_min_len > 1"
                        metrics.update({"label_min_len": label_min_len, "label_max_len": label_max_len, "unlabel_min_len": unlabel_min_len, "unlabel_max_len": unlabel_max_len})

                        # breakpoint()


                    

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    
                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_correction_and_add_to_batch,
                        maybe_apply_rollout_correction,
                    )

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    need_recomputation = maybe_apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                    if need_recomputation:
                        # LEGACY MODE: Compute old_log_probs from actor
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'


                    

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction weights centrally (once per batch)
                        # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
                        # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
                        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            # use_softmax=self.config.trainer.use_softmax,
                            # temperature=self.config.trainer.softmax_temperature,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if self.config.trainer.train_mode != "supervised":
                    metrics.update(compute_data_metrics_semi(batch=batch, use_critic=self.use_critic))
                else:
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


