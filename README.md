# Qwen2-VL-nutrition-table-detection

## Introduction

## Installations:
```
pip install -q torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q -U transformers==4.47.0 trl qwen-vl-utils datasets bitsandbytes peft wandb accelerate matplotlib IPython pydantic Pillow==11.2.1 
```
```
## Install FlashAttention-2 (Optional):
pip install flash-attn
```

## Finetune, and evaluate Qwen2-VL-Instruct model for nutrition-table detection:
1. Finetune using QLoRA:
   ```
   bash launch_finetune_qlora_nutrition_table_detection.sh
   ```
2. Evaluate finetuned model:
   ```
   # --use_quantized_model \
   # --qlora_model_id "<QLORA_MODEL_ID>" \
   
   python evaluate_nutrition_table_detection_iou.py \
    --base_model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --lora_adapters_id "<LORA_ADAPTERS_ID>" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --max_new_tokens 512 \
    --max_num_img_tokens_limit 5250
   ```
   or use the shell script:
   ```
   bash launch_evaluate_nutrition_table_detection_iou.sh
   ```
3. Run inference:
   ```
   # --use_quantized_model \
   # --qlora_model_id "<QLORA_MODEL_ID>" \
   
   python inference_nutrition_table_detection.py \
    --image_url "<IMAGE_URL>" \
    --base_model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --lora_adapters_id "<LORA_ADAPTERS_ID>" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --max_new_tokens 512 \
    --max_num_img_tokens_limit 5250
   ```
   or use the shell script:
   ```
   bash launch_inference_nutrition_table_detection.sh
   ```
