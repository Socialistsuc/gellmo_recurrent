#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1              # 1个节点
#SBATCH --gres=gpu:1           # 申请1张GPU
#SBATCH --cpus-per-task=4      # 给推理进程分配4核CPU
#SBATCH --time=1:00:00        # 最长运行时间
#SBATCH -D /data/home/wgy/workspace/bin/gellmo/GeLLMO

export CUDA_VISIBLE_DEVICES=0

# 执行推理脚本
srun bash inference.sh mistralai/Mistral-7B-Instruct-v0.3 \
                       NingLab/MuMOInstruct \
                       NingLab/GeLLMO-P6-Mistral \
                       /data/home/wgy/workspace/bin/gellmo/GeLLMO/test \
                       bbbp+drd2+plogp+qed \
                       20 \
                       seen
