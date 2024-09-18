
import os
import sys
import json
import math
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from typing import List
import transformers

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

# add all helper packages, used or not used here
sys.path.append('../../block-world-research/')
sys.path.append('../../block-world-research/bw-correction-dialogues')
sys.path.append('../../block-world-research/bw-correction-annotations')
from src import log_utils
logger = log_utils.get_logger(log_utils.DEBUG)

# import evaluate_annotations
# evaluate_annotations.evaluate_predictions(predictions)
sys.path.append('../')
from bw_modelling import evaluation, debug_utils
import evaluate_bw_results
import evaluate_annotations
import data_utils
import unity_utils


DEVICE = "cuda:0"
EVAL_BATCH_SIZE = 1     # keep it at 1
SEED = 124124


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, entries: List[dict], image_folder, tokenizer, image_processor, model_config):
        self.entries = entries
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        entry = self.entries[index]
        image_file = entry['image']
        # print(entry['conversations'])
        # qs = entry['conversations'][0]['value']
        # input(entry.)
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()

        for msg in entry['conversations'][:-1]:
            conv.append_message(conv.roles[0] if msg['from'] == 'human' else conv.roles[1], msg['value'])

        # assert len(conv.roles) == 2, f"Cannot handle long conversations yet!"
        # conv.append_message(conv.roles[0], qs)
        # append a final assistant msg
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.entries)


def create_data_loader(entries: List[dict], image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(entries, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    _generated_data_path = os.path.join(os.environ['UNITY_DATA_DIR'], '576p')
    _block_world_data_path = os.path.join(os.environ['UNITY_DATA_DIR'].replace(
        'generated_data', 'datasets'), 'BlockWorld-Random')

    # original_entries = data_utils.BWEntrySet.from_original_entries(
    #     generated_data_path=_generated_data_path,
    #     block_world_path=_block_world_data_path)

    test_dataset = data_utils.get_dataset(args.test_dataset, _generated_data_path, _block_world_data_path)

    # read input test data
    with open(os.path.join('/users/fjc3/sharedscratch/datasets/llava', args.turn_masking, f"{args.test_dataset}-{args.goal}-test.json"), 'r') as f:
        test_data = json.load(f)

    data_loader = create_data_loader(test_data, args.image_folder, tokenizer, image_processor, model.config)
    predictions = []
    all_responses = []

    try:
        for (input_ids, image_tensor, image_sizes), entry in tqdm(zip(data_loader, test_data), total=len(data_loader)):
            idx = entry['id']

            input_ids = input_ids.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            assert len(generated_text) == 1
            generated_text = generated_text[0].strip()

            # try:
            #     img_token_index = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)
            # except ValueError:
            #     print(f"Image token {IMAGE_TOKEN_INDEX} not found in entry {idx} in input_ids: {input_ids}")
            #     decoded_input_prompt = 'ERROR! ' + tokenizer.decode(input_ids[0])
            # else:
            decoded_input_prompt = debug_utils.decode_input_ids(input_ids[0], tokenizer, handle_image_token=IMAGE_TOKEN_INDEX)
            if len(predictions) == 0:
                # first run, display some info
                # find index with IMAGE_TOKEN_INDEX, as we cannot decode that one
                logger.info(f"Sample: {entry}")
                logger.info(f"Input prompt: \n```{decoded_input_prompt}```")
                logger.info(f"Generated text: \n```{generated_text}```")

            predictions.append(data_utils.Idefics2Prediction.from_model_output(
                test_dataset.get_entry_by_idx(idx), generated_text, args.goal))
            all_responses.append({
                'entry_idx': idx,
                'image_name': entry['image'],
                'input_messages': entry['conversations'],
                'input_prompt': decoded_input_prompt,
                # 'input_labels': debug_utils.decode_label_ids(labels[index], processor=processor),
                'model_response': generated_text,
                'true_answer': entry['answer']
            })

            # evaluation.print_evaluation_table(
            #     evaluation.evaluate_predictions(predictions, True))

            logger.debug(evaluation.get_training_evaluation_str(predictions, total_entries=len(data_loader)))

    except KeyboardInterrupt:
        logger.info("Interrupted by user, saving the results so far")

    evaluation.print_evaluation_table(evaluation.evaluate_predictions(predictions, include_subsets_by_type=True))
    if args.model_base is None:
        args.model_base = model_name
    output_file = evaluation.save_model_outputs(
        info=args.__dict__,
        responses=all_responses)
    logger.info(f"Responses saved to '{output_file}',")
    logger.info(args)
    evaluation.eval_output_file(output_file, test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load", type=str, default="lora", choices=['qlora', 'lora'])
    parser.add_argument("--model_task", type=str, required=True, choices=['zeroshot', 'finetune', 'fewshot-1', 'fewshot-2', 'fewshot-3', 'fewshot-4', 'fewshot-5'])
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--goal", type=str, required=True, choices=['source', 'target'])
    parser.add_argument("--turn_masking", type=str, choices=['none', 'assistant', 'all'], default='none')
    parser.add_argument("--prompt_task_instruction", type=str, choices=['system', 'user'], default='system')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    # args.goal = 'source' if 'source' in args.question_file else 'target'

    logger.info(f"{'*' * 10} Starting eval_llama.py {'*' * 10}")
    logger.info(f"   Run parameters: {args}")

    transformers.set_seed(args.seed)

    if 'checkpoint' in args.model_path:
        logger.info(f"Loading fine-tuned model: {args.model_path}")
    else:
        logger.info(f"Zero-shot evaluation with model: {args.model_path}")

    main(args)

    logger.info(f"{'=' * 10} Finished eval_llama.py {'=' * 10}")
