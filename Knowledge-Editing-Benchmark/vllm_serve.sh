#!/bin/bash
#SBATCH --job-name=know_vserve      
#SBATCH --partition=LMreason      
#SBATCH --nodes=1                  
#SBATCH --gres=gpu:1           
#SBATCH --ntasks-per-node=1       
#SBATCH -o logs/%x_%j.log    


# wyren
# export CUDA_VISIBLE_DEVICES=5

NODE_IP=$(hostname -I | awk '{print $1}')
VLLM_PORT=21910

# # knowledge
# MODEL_PATH="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_mylasong/models/llm_qwen25_72b_instruct/Qwen25-72B-Instruct"
# MODEL_PATH="/home/wyren/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"

# wyren:换成7b跑通pipeline
# MODEL_PATH="/home/wyren/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH="/root/autodl-tmp/model/qwen2.5-7b-instruct"




# # # 进入 vLLM 项目目录
# cd /mnt/petrelfs/xuxingcheng/VLMReasoning/RLTraining_backup/openrlhf_v/eval

# source activate ws_openrlhf_qwen25vl

echo "Starting vLLM service on ${NODE_IP}:${VLLM_PORT}..."

# 启动 vLLM 服务器
vllm serve $MODEL_PATH \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --served-model-name Qwen2.5-VL \
    --max-model-len 8192 \
    --dtype bfloat16 &

# 等待vLLM服务启动
echo "Waiting for vLLM service to start..."
sleep 10
while ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; do
    sleep 10
    echo "Still waiting for vLLM service..."
done
echo "vLLM service is ready!"

wait