# if you have ``.metadata`` in the fsdp checkpoint path, directly use ``accelerate merge-weights <ckpt_path> <target_path>`` in terminal
# otherwise, you have to resume the original training setting, say if you use 4 gpu on a single node, launch as below
# torchrun --nnodes 1 --nproc-per-node 4 convert_fsdp_to_hf_without_meta.py
# the rationale behind this is load the weight, find the device mesh, then build the FSDP wrap based on the device mesh


import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, StateDictType, ShardedStateDictConfig
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '8'

def load_fsdp_and_save(config_path, tokenizer_path, ckpt_path, target_path):
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    local_model_path = os.path.join(ckpt_path, f'model_world_size_{world_size}_rank_{rank}.pt')
    print(f'[rank-{rank}]: Loading from {local_model_path}')
    model_state_dict = torch.load(local_model_path, weights_only=False)
    saved_device_mesh = model_state_dict[list(model_state_dict.keys())[0]].device_mesh
    device_mesh = dist.init_device_mesh(saved_device_mesh.device_type, mesh_shape=saved_device_mesh.mesh.shape, mesh_dim_names=saved_device_mesh.mesh_dim_names)
    
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    config = AutoConfig.from_pretrained(config_path)
    with torch.device('cuda'):
        hf_model = AutoModelForCausalLM.from_config(config=config,
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation='flash_attention_2')
        hf_model = hf_model.to(device='cuda')
    
    # Wrap HF model with FSDP
    fsdp_model = FSDP(hf_model,
                use_orig_params=True,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                device_mesh=device_mesh)
    fsdp_model.eval()
    
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
        fsdp_model.load_state_dict(model_state_dict)    

    # save
    with FSDP.summon_full_params(fsdp_model):
        fsdp_model.module.save_pretrained(target_path, max_shard_size="4GB")
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(target_path)
    dist.barrier()

def check_save(target):
    model = AutoModelForCausalLM.from_pretrained(target, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target)
    
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    
    
if __name__ == "__main__":
    ckpt_path = "/ossfs/workspace/aml0/484999/code/LUFFY/luffy/verl/checkpoints/LUFFY_Qwen_7B_DM_1k_AUG_cft_mix_tmp_unfix/global_step_20/actor"
    tokenizer_path = "/ossfs/workspace/aml0/484999/code/LUFFY/luffy/verl/checkpoints/LUFFY_Qwen_7B_DM_1k_AUG_cft_mix_tmp_unfix/global_step_20/actor/huggingface/tokenizer.json"
    config_path = "/ossfs/workspace/aml0/484999/code/LUFFY/luffy/verl/checkpoints/LUFFY_Qwen_7B_DM_1k_AUG_cft_mix_tmp_unfix/global_step_20/actor/huggingface/config.json"
    target_path = "/ossfs/workspace/aml0/484999/code/LUFFY/luffy/verl/checkpoints/LUFFY_Qwen_7B_DM_1k_AUG_cft_mix_tmp_unfix/merge/actor_20"
    load_fsdp_and_save(config_path, tokenizer_path, ckpt_path, target_path)
    # check corectness by comment above statement and uncomment below statement, then run ``python convert_fsdp_tp_hf_without_meta.py``
    # check_save(target_path)