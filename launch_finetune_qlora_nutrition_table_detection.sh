#!/bin/bash

# --calculate_loss_on_completions_only \
# --train_existing_adapters \
# --existing_lora_adapters_id "atk-25/Qwen2-VL-7B-Instruct-nutrition-table-detection-LoRA-adapters"\

python finetune_qlora_nutrition_table_detection.py \
    --base_model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --attn_implementation "flash_attention_2" \
    --save_local \
    --save_local_adapters \
    --push_to_hub \
    --push_to_hub_adapters \
    --create_new_repo \
    --create_new_repo_adapters \
    --push_to_hub_repo_id "<REPO_ID>" \
    --push_to_hub_repo_id_adapters "<REPO_ID_ADAPTERS>" \
    --repos_private \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_seq_length 5626 \
    --num_train_epochs 1 \
    --max_steps -1 \
    --learning_rate 2e-4 \
    --optim "adamw_torch_fused" \
    --lr_scheduler_type "linear" \
    --bf16 \
    --tf32 \
    --warmup_steps 10 \
    --weight_decay 0.001 \
    --logging_steps 2 \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --save_strategy "steps" \
    --save_steps 30 \
    --report_to "wandb" \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --output_dir "Qwen2-VL-7B-Instruct-nutrition-table-detection-qlora" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --WANDB_PROJECT "Qwen2-VL-7B-Instruct-nutrition-table-detection" \
    --WANDB_NAME "Qwen2-VL-7B-Instruct-nutrition-table-detection-qlora" \
    --max_new_tokens 512 \
    --max_num_img_tokens_limit 5250
