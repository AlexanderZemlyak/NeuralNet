#!bash
deepspeed --num_gpus 2 finetune_deepseekcoder2_normal+enriched.py \
    --model_name_or_path deepseek-ai/deepseek-coder-1.3b-instruct \
    --data_path ./train_data_normal+enriched.json \
    --output_dir ./LoRa \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 25 \
    --learning_rate 2e-4 \
    --warmup_steps 30 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --bf16 True \
    --model_max_length 2048 \
    --save_only_model True \
    --deepspeed ./ds_config_zero1_bf16.json