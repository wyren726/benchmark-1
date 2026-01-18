#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=3

# 
# echo "llama 3 IKE Single all..."
# python run_LLM_evaluation.py \
# --editing_method IKE \
# --hparams_dir /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/EasyEdit/hparams/IKE/llama3.1-8b.yaml \
# --data_dir /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/ModelEdit/EasyEdit_new/examples/dataset/zsre/ZsRE-test-all.json \
# --train_data_path /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/ModelEdit/EasyEdit_new/examples/dataset/zsre/zsre_train_10000.json \
# --datatype zsre \
# --ds_size 1319 \
# --start_index 0 \
# --end_index 1319 \
# # --sequential_edit 

echo "llama 3 IKE Single all..."
python run_LLM_evaluation.py \
--editing_method IKE \
--hparams_dir /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/EasyEdit/hparams/IKE/llama3.1-8b.yaml \
--data_dir /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/ModelEdit/EasyEdit_new/examples/dataset/zsre/ZsRE-test-all.json \
--train_data_path /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/ModelEdit/EasyEdit_new/examples/dataset/zsre/zsre_train_10000.json \
--datatype zsre \
--ds_size 1319 \
--start_index 0 \
--end_index 1319 \
# --sequential_edit 

