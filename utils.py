import ast
import gc
import time
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from pydantic import BaseModel
import torch
from torchvision import ops
from qwen_vl_utils import process_vision_info
from datasets import load_dataset


norm_width, norm_height = 1000, 1000


def get_dataset(train_dataset=True, eval_dataset=True):

    dataset_id = "openfoodfacts/nutrition-table-detection"

    if train_dataset:
        train_ds = load_dataset(dataset_id, split="train")
    else:
        train_ds = None

    if eval_dataset:
        eval_ds = load_dataset(dataset_id, split="val")
    else:
        eval_ds = None

    return train_ds, eval_ds


# helper function to plot bbox on an input example from the unprocessed dataset
def bbox_visualize_raw_ds(example):

    image = example["image"]

    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)
    num_bbox = len(example["objects"]["bbox"])   # number of bounding boxes
    colors = ['red', 'green', 'blue']

    for i in range(num_bbox):
        # get relative coordinates of the bbox, and change the scale from 0-1 to 0-1000
        bbox = example["objects"]["bbox"][i]
        rel_x_1 = bbox[1]*norm_width    # x_top_left
        rel_y_1 = bbox[0]*norm_height   # y_top_left
        rel_x_2 = bbox[3]*norm_width    # x_bottom_right
        rel_y_2 = bbox[2]*norm_height   # y_bottom_right

        abs_x_1 = int(rel_x_1/norm_width * im_width)
        abs_y_1 = int(rel_y_1/norm_height * im_height)
        abs_x_2 = int(rel_x_2/norm_width * im_width)
        abs_y_2 = int(rel_y_2/norm_height * im_height)

        bbox = [abs_x_1, abs_y_1, abs_x_2, abs_y_2]
        label = example["objects"]["category_name"][i]
        print(f"label: {label},   bbox: {bbox}")

        color = colors[i%3]
        draw.rectangle((bbox[:2], bbox[2:]), outline=color, width=int(15/2000*im_width))
        draw.text((bbox[0] + 8, bbox[1] + 6), label, fill=color, font_size=int(0.1*(abs_x_2-abs_x_1)))

    plt.imshow(image)
    plt.axis('off')
    plt.show()


# helper function to get bbox from an example of the dataset
def get_bbox_ds_example(image, bbox_ds):
    im_width, im_height = image.size
    rel_x_1 = int(bbox_ds[1] * norm_width)  # x_top_left
    rel_y_1 = int(bbox_ds[0] * norm_height)  # y_top_left
    rel_x_2 = int(bbox_ds[3] * norm_width)  # x_bottom_right
    rel_y_2 = int(bbox_ds[2] * norm_height)  # y_bottom_right

    abs_x_1 = int(rel_x_1 / norm_width * im_width)
    abs_y_1 = int(rel_y_1 / norm_height * im_height)
    abs_x_2 = int(rel_x_2 / norm_width * im_width)
    abs_y_2 = int(rel_y_2 / norm_height * im_height)

    bbox_relative = [rel_x_1, rel_y_1, rel_x_2, rel_y_2]
    bbox_absolute = [abs_x_1, abs_y_1, abs_x_2, abs_y_2]

    return bbox_relative, bbox_absolute


system_message = "You are a Vision Language Model specialized in interpreting visual data from product images. " \
                 "Your task is to analyze the provided product images and detect the nutrition tables, delivering coordinates " \
                 "in bbox format {'bbox_2d': [x1, y1, x2, y2], 'label': 'category name'}. Avoid additional explanation unless absolutely necessary."

prompt = "Detect the bounding boxes of the nutrition tables present in the image. " \
         "Output format for each detected table should be as follows: {'bbox_2d': [x1, y1, x2, y2], 'label': 'category name'}."

def create_conversation_template_for_training(example, system_message, prompt):

    image = example['image']
    num_bboxes = len(example["objects"]["bbox"])
    assistant_response = ''

    for i in range(num_bboxes):
        category_name = example["objects"]["category_name"][i]
        bbox_ds = example["objects"]["bbox"][i]
        bbox_rel, _ = get_bbox_ds_example(image, bbox_ds)
        assistant_response_bbox_i = '{"bbox_2d": ' + f'<|box_start|>{bbox_rel}<|box_end|>' + ', "label": ' f'<|object_ref_start|>"{category_name}"<|object_ref_end|>' + '}, '
        assistant_response += assistant_response_bbox_i

    assistant_response = '[' + assistant_response[:-2] + ']'

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_response,
                }
            ],
        },
    ]

    return messages


# helper function to determine the maximum pixel size
def calculate_max_num_img_tokens(images, patch_size, merge_size):

    image_sizes = []
    for image in images:
        image_size = image[0].size[0] * image[0].size[1]
        image_sizes.append(image_size)

    max_image_size = max(image_sizes)
    max_num_image_tokens = int( max_image_size / (patch_size*patch_size) / (merge_size*merge_size) )
    return max_num_image_tokens



# inference function for object detection
def nutrition_table_detection_inference(model, processor, device, image_url, system_message, prompt, max_new_tokens=512):

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return image_inputs[0], output_text[0]


# class to validate model output format using pydantic BaseModel class
class ValidateModelOutputFormat(BaseModel):
    bbox_2d: list[int]
    label: str


# function to parse model output to extract bounding box coordinates and convert the relative coordinates to absolute coordinates
def parse_bbox_model_output(image, output_text, image_idx=0):
    predicted_tables = []
    try:
        output_text_parsed = ast.literal_eval(output_text)

        im_width, im_height = image.size

        if isinstance(output_text_parsed, dict):
            output_text_parsed = [output_text_parsed]

        skipped_tables_count = 0
        for i in range(len(output_text_parsed)):
            try:
                ValidateModelOutputFormat.model_validate(output_text_parsed[i])   # validate model output format
                bbox_relative = output_text_parsed[i]['bbox_2d']
                abs_x_1 = int(bbox_relative[0] / norm_width * im_width)
                abs_y_1 = int(bbox_relative[1] / norm_width * im_height)
                abs_x_2 = int(bbox_relative[2] / norm_width * im_width)
                abs_y_2 = int(bbox_relative[3] / norm_width * im_height)
                bbox_absolute = [abs_x_1, abs_y_1, abs_x_2, abs_y_2]
                output_text_parsed[i]['bbox_2d'] = bbox_absolute
                predicted_tables.append(output_text_parsed[i])
            except Exception as e:
                print(f"Validation Error: {e}")
                skipped_tables_count += 1

            if skipped_tables_count > 0:
                print(f"image[{image_idx}]: {skipped_tables_count} predicted table(s) skipped since the model output for these tables is not in the expected format")
    except Exception as e:
        print(f"Validation Error: {e}")
        print(
            f"image[{image_idx}]: the model output for this image can not be parsed into the expected format. ---> predicted_tables = []")

    return predicted_tables


def bbox_visualize_predicted_tables(image, predicted_tables):
    colors = ['red', 'green', 'blue']
    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)

    for i in range(len(predicted_tables)):
        color = colors[i % 3]
        label = predicted_tables[i]['label']
        bbox_absolute = predicted_tables[i]['bbox_2d']
        draw.rectangle((bbox_absolute[:2], bbox_absolute[2:]), outline=color, width=int(15 / 2000 * im_width))
        draw.text((bbox_absolute[0] + 8, bbox_absolute[1] + 6), label, fill=color,
                  font_size=int(0.1 * (bbox_absolute[2] - bbox_absolute[0])))

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def evaluate_model_iou(model, processor, device, dataset, system_message, prompt, max_new_tokens=512):

    if isinstance(dataset['image_id'], str) == True:
        dataset['image'] = [dataset['image']]
        dataset["objects"] = [dataset["objects"]]

    iou_all = []
    for i in range(len(dataset['image'])):  # iterate through all examples

        image_url = dataset['image'][i]
        category_names = dataset["objects"][i]["category_name"]  # get category names of the tables present in the image
        bboxes_ds = dataset["objects"][i]["bbox"]  # get bounding boxes of the categories present in the image

        ground_truth_tables = []
        for k in range(len(bboxes_ds)):  # iterate through all the tables in the image and create a dictionary for each table
            _, bbox_ground_truth_abs = get_bbox_ds_example(image_url, bboxes_ds[k])
            bbox_ground_truth_abs = torch.tensor(bbox_ground_truth_abs)
            ground_truth_tables.append(
                {"category_name": category_names[k], "bbox_ground_truth_abs": bbox_ground_truth_abs,
                 "bbox_predicted_abs": None, "iou": 0})

        image, output_text = nutrition_table_detection_inference(model, processor=processor, device=device, image_url=image_url,
                                                                 system_message=system_message, prompt=prompt,
                                                                 max_new_tokens=max_new_tokens)
        predicted_tables = parse_bbox_model_output(image, output_text, image_idx=i)

        # For each ground truth table in the image, calculate iou's with respect to each of the detected tables of the same category name.
        # The predicted table with the largest iou is selected to be the predicted bbox for it's corresponding ground truth table.
        # If all the iou's are zero, then it is assumed that the current table is not detected and an iou=0 is used for calculating average iou.

        for ground_truth_table in ground_truth_tables:
            iou_max = 0
            for predicted_table in predicted_tables:
                if predicted_table["label"] == ground_truth_table["category_name"]:
                    bbox_ground_truth_abs = ground_truth_table["bbox_ground_truth_abs"].reshape(1, 4)
                    bbox_predicted_abs = torch.tensor(predicted_table["bbox_2d"]).reshape(1, 4)
                    iou = ops.box_iou(bbox_ground_truth_abs, bbox_predicted_abs)
                    if iou > iou_max:
                        ground_truth_table["bbox_predicted_abs"] = bbox_predicted_abs
                        ground_truth_table["iou"] = iou
                        iou_max = iou

        iou_per_image_all = []
        for ground_truth_table in ground_truth_tables:
            iou_per_image_all.append(ground_truth_table["iou"])
        iou_per_image_ave = sum(iou_per_image_all) / len(iou_per_image_all)

        iou_all.append(iou_per_image_ave)

    return (sum(iou_all) / len(iou_all)).item()


def clear_memory():
    if 'train_ds' in globals(): del globals()['train_ds']
    if 'eval_ds' in globals(): del globals()['eval_ds']
    if 'train_ds_raw' in globals(): del globals()['train_ds_raw']
    if 'eval_ds_raw' in globals(): del globals()['eval_ds_raw']
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    if 'model_q_nf4' in globals(): del globals()['model_q_nf4']
    if 'qlora_model' in globals(): del globals()['qlora_model']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

