#!/bin/bash
# --use_quantized_model \
# --qlora_model_id "<QLORA_MODEL_ID>" \

python evaluate_nutrition_table_detection.py \
    --base_model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --lora_adapters_id "LORA_ADAPTERS_ID" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --max_new_tokens 512 \
    --max_num_img_tokens_limit 5250
