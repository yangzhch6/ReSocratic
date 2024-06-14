export MODEL_NAME='llama2-7b-chat'
export DATA_NAME="resocratic-29k"
export MODEL_PATH=""  ### specify uour model path
export PROJ_NAME="${MODEL_NAME}-3ep-${DATA_NAME}" 
export SAVE_PATH="checkpoints/${PROJ_NAME}" 
export TRAIN_DATA_PATH="synthesis_data/${DATA_NAME}.json"

export MASTER_ADDR="localhost"
export MASTER_PORT="1236"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_PROJECT=${PROJ_NAME}
wandb online

echo $TRAIN_DATA_PATH
echo $SAVE_PATH
echo "------------------------------------------------------------------------------"

python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=2 --use_env train_llama2_code.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 7 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --prim_mode False 