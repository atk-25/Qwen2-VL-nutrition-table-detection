from utils import *

import os
import argparse
import logging
import torch
from huggingface_hub import HfApi, login
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from qwen_vl_utils import vision_process
from qwen_vl_utils import process_vision_info
from transformers.utils import is_flash_attn_2_available
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
import wandb


logger = logging.getLogger(__name__)

norm_width, norm_height = 1000, 1000


def arg_parser():

    parser = argparse.ArgumentParser()

    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    parser.add_argument("--base_model_id", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=attn_implementation)
    parser.add_argument("--save_local", action="store_true")
    parser.add_argument("--save_local_adapters", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--push_to_hub_adapters", action="store_true")
    parser.add_argument("--create_new_repo", action="store_true")
    parser.add_argument("--create_new_repo_adapters", action="store_true")
    parser.add_argument("--train_existing_adapters", action="store_true")
    parser.add_argument("--existing_lora_adapters_id", type=str, default=None)
    parser.add_argument("--push_to_hub_repo_id", type=str, default=None)
    parser.add_argument("--push_to_hub_repo_id_adapters", type=str, default=None)
    parser.add_argument("--repos_private", action="store_true")

    ### SFTConfig related arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_kwargs", type=dict, default={"use_reentrant": False})
    parser.add_argument("--max_seq_length", type=int, default=5000)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.001)

    # logging parameters
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--output_dir", type=str, default="Checkpoints_adapters")

    parser.add_argument("--dataset_kwargs", type=dict, default={"skip_prepare_dataset": True})

    parser.add_argument("--calculate_loss_on_completions_only", action="store_true")

    # PEFT related parameters
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")

    # wandb project and name associated with this run
    parser.add_argument("--WANDB_PROJECT", type=str, default=None)
    parser.add_argument("--WANDB_NAME", type=str, default=None)

    # max_new_tokens:   used for evaluation function
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--max_num_img_tokens_limit", type=int, default=5000)

    args = parser.parse_args()

    return args


def get_model_and_processor_qlora(args):

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

    model_q_nf4 = prepare_model_for_kbit_training(
        model_q_nf4,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
    )

    if args.train_existing_adapters:
        model = PeftModel.from_pretrained(model_q_nf4, args.existing_lora_adapters_id)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj"],   # attach adapters to attention layers of the Language model and vision encoder
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model_q_nf4, lora_config)

    model.print_trainable_parameters()

    return model, processor


def collate_fn(examples):

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    images = [process_vision_info(example)[0] for example in examples]

    # Get batch inputs to LLM
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)   # a list of dictionaries with these keys ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

    # Get the labels to the input_ids
    labels = batch["input_ids"].clone()
    # Mask the padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100  # pad_token_id = 151643

    ## Mask tokens related to image
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)   # image_token_id = 151655
    # Also include tokens that represent the start and end of image tokens
    image_tokens = [image_token_id] + [151652, 151653]   # <|vision_start|> = 151652, <|vision_end|> = 151653

    for image_token in image_tokens:
        labels[labels == image_token] = -100

    # add the labels to the input batch
    batch["labels"] = labels   # batch: a list of dictionaries with these keys ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'labels']

    return batch


def collate_fn_completions_only_loss(examples):

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    images = [process_vision_info(example)[0] for example in examples]   # process the images and keep only the image_inputs (video_inputs is 'None' since there are no videos in the ds)

    # Get batch inputs to LLM
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)   # a list of dictionaries with these keys ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

    # Get the labels to the input_ids
    labels = batch["input_ids"].clone()

    # Mask all tokens except the completion (assistant response) tokens
    for i in range(labels.shape[0]):
        # vision_start_token_id = 151652
        # vision_start_idx = labels[i].tolist().index(vision_start_token_id)
        vision_end_token_id = 151653
        vision_end_idx = labels[i].tolist().index(vision_end_token_id)
        im_end_token_id = 151645
        prompt_end_idx = labels[i].tolist().index(im_end_token_id, vision_end_idx)
        labels[i][ : prompt_end_idx+2] = -100   # [ : prompt_end_idx+2] to also mask the new line token after "im_end" of the prompt sequence

    # add the labels to the input batch
    batch["labels"] = labels   # batch: a list of dictionaries with these keys ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'labels']

    return batch


def finetune_qlora(model, processor, train_ds, eval_ds, args, collate_fn):

    sft_config = SFTConfig(

        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing = args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
        max_seq_length = args.max_seq_length,

        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,

        bf16=args.bf16,
        tf32=args.tf32,

        warmup_steps = args.warmup_steps,
        weight_decay = args.weight_decay,

        # logging parameters
        logging_steps = args.logging_steps,
        eval_strategy = args.eval_strategy,
        eval_steps = args.eval_steps,
        save_strategy = args.save_strategy,
        save_steps = args.save_steps,
        report_to = args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        output_dir = args.output_dir,

        dataset_kwargs=args.dataset_kwargs,
        label_names=["labels"],

    )

    ### set up wandb.init
    wandb.init(
        project = args.WANDB_PROJECT,
        name = args.WANDB_NAME,
        config = sft_config,
    )

    trainer = SFTTrainer(
        model = model,
        args = sft_config,
        data_collator=collate_fn,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        processing_class = processor,
    )

    print("Training Started.")

    trainer.train()

    print("Training completed.")


if __name__ == "__main__":

    args = arg_parser()

    if args.repos_private:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if the model_id repo is private"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.push_to_hub or args.create_new_repo:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if you want to push model to a HF repo"
        assert args.push_to_hub_repo_id is not None, "push_to_hub_repo_id is not provided"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.push_to_hub_adapters or args.create_new_repo_adapters:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if you want to push lora adapters to a HF repo"
        assert args.push_to_hub_repo_id_adapters is not None, "push_to_hub_repo_id_adapters is not provided"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.save_local or args.save_local_adapters:
        assert args.output_dir is not None, "output_dir is not provided"

    if args.train_existing_adapters:
        assert args.existing_lora_adapters_id is not None, "existing_lora_adapters_id is not provided"

    if args.report_to=='wandb':
        assert os.environ.get('WANDB_API_KEY') is not None, "Set up WANDB_API_KEY in order to report to 'wandb'"
        WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
        wandb.login(key=WANDB_API_KEY)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model_and_processor_qlora(args=args)

    ### limit max number of image tokens & GPU memory usage by limiting image MAX_PIXELS
    patch_size = 14   # model.base_model.model.visual.config.patch_size
    merge_size = 2    # processor.image_processor.merge_size

    # images = [process_vision_info(example)[0] for example in train_ds]
    # max_num_image_tokens = calculate_max_num_img_tokens(images, patch_size, merge_size)
    # print(max_num_image_tokens)   # calculated to be 16320, which is close to the default 16384

    max_num_img_tokens_limit = args.max_num_img_tokens_limit
    max_pixels = max_num_img_tokens_limit * (patch_size * patch_size) * (merge_size * merge_size)
    vision_process.MAX_PIXELS = max_pixels

    ### get dataset
    train_ds_raw, eval_ds_raw = get_dataset(train_dataset=True, eval_dataset=True)

    system_message = "You are a Vision Language Model specialized in interpreting visual data from product images. " \
                     "Your task is to analyze the provided product images and detect the nutrition tables, delivering coordinates " \
                     "in bbox format {'bbox_2d': [x1, y1, x2, y2], 'label': 'category name'}. Avoid additional explanation unless absolutely necessary."

    prompt = "Detect the bounding boxes of the nutrition tables present in the image. " \
             "Output format for each detected table should be as follows: {'bbox_2d': [x1, y1, x2, y2], 'label': 'category name'}."

    train_ds = [create_conversation_template_for_training(example, system_message=system_message, prompt=prompt) for example in train_ds_raw]
    eval_ds = [create_conversation_template_for_training(example, system_message=system_message, prompt=prompt) for example in eval_ds_raw]

    ### finetune the model using qlora method
    if args.calculate_loss_on_completions_only:
        finetune_qlora(model, processor=processor, train_ds=train_ds, eval_ds=eval_ds, args=args, collate_fn=collate_fn_completions_only_loss)
    else:
        finetune_qlora(model, processor=processor, train_ds=train_ds, eval_ds=eval_ds, args=args, collate_fn=collate_fn)

    ### save adapters
    if args.save_local_adapters:
        model.save_pretrained(args.output_dir + "-adapters")
        logger.info(f"lora adapters saved to local directory.")

    if args.create_new_repo_adapters:
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub_repo_id_adapters, private=True)
        logger.info(f"New repo created for pushing lora adapters, repo_id: {args.push_to_hub_repo_id_adapters}")

    if args.push_to_hub_adapters:
        processor.push_to_hub(args.push_to_hub_repo_id_adapters)
        model.push_to_hub(args.push_to_hub_repo_id_adapters)
        logger.info(f"lora adapters and processor pushed to repo_id: {args.push_to_hub_repo_id_adapters}")

    # evaluate finetuned model using Intersection Over Union (IOU) metric.
    iou = evaluate_model_iou(model, processor=processor, device=device, dataset=eval_ds_raw, system_message=system_message,
                             prompt=prompt, max_new_tokens=args.max_new_tokens)
    logger.info(f"iou score, over eval_dataset:   {iou:.2f}")
    print(f"iou score, over eval_dataset:   {iou:.2f}")

    model = model.merge_and_unload()

    # Save merged model
    if args.save_local:
        model.save_pretrained(args.output_dir + "-merged")
        print(f"merged qlora model saved to local directory.")

    if args.create_new_repo:
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub_repo_id, private=True)
        logger.info(f"New repo created for pushing merged qlora model, repo_id: {args.push_to_hub_repo_id}")
        print(f"New repo created for pushing merged qlora model, repo_id: {args.push_to_hub_repo_id}")

    if args.push_to_hub:
        processor.push_to_hub(args.push_to_hub_repo_id)
        model.push_to_hub(args.push_to_hub_repo_id)
        logger.info(f"merged qlora model and processor pushed to repo_id: {args.push_to_hub_repo_id}")
        print(f"merged qlora model and processor pushed to repo_id: {args.push_to_hub_repo_id}")

    clear_memory()
