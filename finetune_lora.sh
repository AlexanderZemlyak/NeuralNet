#!bash
python finetune_deepseekcoder2.py \
    --model_name_or_path deepseek-ai/deepseek-coder-1.3b-instruct \
    --data_path ./train_data.json \
    --output_dir ./LoRa \
    --num_train_epochs 10 \
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
    --model_max_length 1024