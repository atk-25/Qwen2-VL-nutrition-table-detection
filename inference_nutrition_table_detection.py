from utils import *

import os
import argparse
import logging
import torch
from huggingface_hub import login
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import vision_process
from transformers.utils import is_flash_attn_2_available
from peft import PeftModel

logger = logging.getLogger(__name__)

norm_width, norm_height = 1000, 1000


def arg_parser():

    parser = argparse.ArgumentParser()

    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    parser.add_argument("--image_url", type=str, default=None)

    parser.add_argument("--base_model_id", type=str, default=None)
    parser.add_argument("--lora_adapters_id", type=str, default=None)
    parser.add_argument("--use_quantized_model", action="store_true")
    parser.add_argument("--qlora_model_id", type=str, default=None)
    parser.add_argument("--repos_private", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=attn_implementation)

    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--max_num_img_tokens_limit", type=int, default=5000)

    args = parser.parse_args()

    return args


def get_model_and_processor(args, device):

    # download Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
    )
    processor = Qwen2VLProcessor.from_pretrained(args.base_model_id)

    model = PeftModel.from_pretrained(model, args.lora_adapters_id)
    # merge the (quantized) base model and adapter for inference
    model = model.merge_and_unload()

    return model, processor


def get_model_and_processor_qlora(args, device):

    if args.qlora_model_id is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qlora_model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
        )
        processor = Qwen2VLProcessor.from_pretrained(args.qlora_model_id)
        return model, processor

    else:
        bnb_nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # load an NF4 model
        model_q_nf4 = Qwen2VLForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_nf4_config,
            use_flash_attention_2=True,    # attn_implementation=args.attn_implementation,
        )

        processor = Qwen2VLProcessor.from_pretrained(args.base_model_id)

        model = PeftModel.from_pretrained(model_q_nf4, args.lora_adapters_id)
        # merge the (quantized) base model and adapter for inference
        model = model.merge_and_unload()

        return model, processor


if __name__ == "__main__":

    args = arg_parser()

    if args.repos_private:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if the model_id repo is private"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.use_quantized_model==False or (args.use_quantized_model==True and args.qlora_model_id is None):
        assert args.base_model_id is not None, "base_model_id is not provided"
        assert args.lora_adapters_id is not None, "lora_adapters_id is not provided"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model and processor
    if args.use_quantized_model:
        model, processor = get_model_and_processor_qlora(args, device)
    else:
        model, processor = get_model_and_processor(args, device)

    ### limit max number of image tokens & GPU memory usage by limiting image MAX_PIXELS
    patch_size = 14   # model.base_model.model.visual.config.patch_size
    merge_size = 2    # processor.image_processor.merge_size

    max_num_img_tokens_limit = args.max_num_img_tokens_limit
    max_pixels = max_num_img_tokens_limit * (patch_size * patch_size) * (merge_size * merge_size)
    vision_process.MAX_PIXELS = max_pixels

    system_message = "You are a Vision Language Model specialized in interpreting visual data from product images. " \
                     "Your task is to analyze the provided product images and detect the nutrition tables in a certain format. " \
                     "Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."
    
    prompt = "Detect the bounding boxes of nutrition tables present in the image. " \
             "Deliver coordinates in bbox format {'bbox_2d': [x1, y1, x2, y2], 'label': 'category name'}."

    image = Image.open(args.image_url)

    output_text = nutrition_table_detection_inference(model, processor=processor, device=device, image=image,
                                                             system_message=system_message, prompt=prompt, max_new_tokens=args.max_new_tokens)
    predicted_tables = parse_bbox_model_output(image, output_text)
    bbox_visualize_predicted_tables(image, predicted_tables)
    plt.savefig(f"annotated_{args.image_url}")

    clear_memory()
