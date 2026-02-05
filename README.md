<div align="center">


<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;"> 
 <!-- <img src="./figures/logo_trapo.png" alt="LUFFY Icon" width="120"> -->
  TRAPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning.
</h1>
<p align="center"><em>TRAPO is A Friend : )</em></p>

<div align="center">
  <img src="./figures/logo_trapo.png" alt="overview" style="width: 33%; height: auto;">
</div>




[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.13106) [![Github](https://img.shields.io/badge/TRAPO-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ShenzhiYang2000/TRAPO)   






</div>

---

# 📚 Overview
- 🎉 [News](#news)  
- 📖 [Introduction](#introduction)  
- ✨ [Getting Started](#getting-started)  
- 🔧 [Usage](#usage)  
- 📃 [Evaluation](#evaluation)  
- 🎈 [Citation](#citation)  
- 🌻 [Acknowledgement](#acknowledgement)  
<!-- - 📈 [Star History](#star-history) -->


---


# 🎉News
<!-- - **[2025/05/30]** We integrate the implementation and scripts of **other off-policy learning methods** including SFT, SFT+RL and RL w/ SFT Loss (multi-task learning).
- **[2025/05/21]** We have updated the paper [version](https://arxiv.org/abs/2504.14945), which re-evaluates all models using a more accurate verifier and adds comparisons with other off-policy learning methods, including RL with SFT Loss and SFT+RL.
- **[2025/04/23]** Our paper now trending on [alphaXiv](https://www.alphaxiv.org/abs/2504.14945)! We welcome feedback and discussion.
- **[2025/04/23]** 🎉 Ranked **#1** of the day on [Huggingface Daily Papers](https://huggingface.co/papers/2504.14945).
- **[2025/04/20]** LUFFY paper available on [arXiv](http://arxiv.org/abs/2504.14945).  -->

<!-- - **[2025/04/20]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4). -->
- **[2025/12/10]** TRAPO codebase is released along with evaluation scripts. Try it out!

---

# 📖Introduction

<!-- ### Core assumption:
 ```When an unlabeled sample is correctly understood (i.e., its pseudo-label is accurately estimated), its learning dynamics will align consistently with those of labeled samples.``` -->

TraPO is a **semi-supervised** reinforcement learning framework that bridges unlabeled and labeled samples for training large reasoning models (LRMs).

<div align="center">
  <img src="./figures/diff_framework.png" alt="overview" style="width: 50%; height: auto;">
</div>

Built upon GRPO, TraPO leverages a small set of labeled examples to guide training on unlabeled data, ensuring that **only reasoning patterns verified on labeled instances are reinforced**. Its core component identifies reliable unlabeled samples by matching their learning trajectories to those of labeled ones, stabilizing consistency-based training and mitigating model collapse caused by unchecked self-reinforcement.

<div align="center">
  <img src="./figures/trapo_notebooklm.png" alt="overview" style="width: 66%; height: auto;">
</div>

<!-- ![overview](./figures/main_results.png) -->






---

# ✨Getting Started

## Installation

You can install TRAPO dependencies by running the following commands:
```bash
conda create -n trapo python=3.10
conda activate trapo
cd TraPO
pip install -r requirements.txt
pip install -e .
pip3 install -e .[vllm]
pip3 install math-verify==0.8.0
```

## Repo Structure

This repository includes:

- `Weakly_Supervised`: **[Recommended]** In this folder, we provide a more flexible and concise implementation interface for TraPO.
- `initial/original_TraPO`: Codes for reproduce our experiment results in the paper.

---





# 🔧Usage

## Data Preparation
To run TraPO for semi-supervised RLVR training, you only need to annotate whether each sample is labeled or not in the `extra_info` field, in addition to your regular data. For details, please refer to the file `./data/code/merge_and_label.py`.

For the data required to reproduce the experiments in the paper, please follow the instructions in `initial/original_TraPO/README.md`.





## Training

We provide an example script to train TraPO in `Weakly_Supervised/scripts/run`:

```bash
  bash run_semi.sh
```


</details>


## Models
The model weights will be uploaded to HuggingFace later.


---

# 📃Evaluation

## Reproducing the Results 
We currently support automated evaluation on six widely used mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro).

You can reproduce our results by running the following commands:
```bash
ROOT=YOUR_ROOT_PATH
DATA=$ROOT/data/valid.math.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=your_model_path
MODEL_NAME=luffy

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
```

# Citation
If you find our code useful, please kindly cite our paper:
```bib
@article{yang2025trapo,
  title={TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning},
  author={Yang, Shenzhi and Zhu, Guangcheng and Zheng, Xing and MA, Yingfan and Chen, Zhongqi and Song, Bowen and Wang, Weiqiang and Zhao, Junbo and Chen, Gang and Wang, Haobo},
  journal={arXiv preprint arXiv:2512.13106},
  year={2025}
}
```


# 🌻Acknowledgement

TRAPO builds upon [LUFFY](https://github.com/ElliottYan/LUFFY), [veRL](https://github.com/volcengine/verl) and [deepscaler](https://github.com/agentica-project/rllm), and utilizes [vLLM](https://github.com/vllm-project/vllm) for inference. We utilize [Math-Verify](https://github.com/huggingface/Math-Verify) for math reasoning evaluation. We thank the open-source community for datasets and backbones, including [DeepMath](https://arxiv.org/abs/2504.11456), [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [DeepSeek-R1](https://github.com/deepseek-ai/deepseek-r1) model. Lastly, we would like to express our gratitude to [NotebookLM](https://notebooklm.google/) for creating the illustrative diagram of TRAPO.

# 📬 Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:
- Shenzhi Yang: yangshenzhi@zju.edu.cn



