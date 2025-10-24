from typing import List, Dict, Union
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple
from collections import Counter

from llava.mm_utils import (tokenizer_image_token)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates


def find_word_spans(
    text: str,
    terms: Iterable[str],
    *,
    case_sensitive: bool = False,
    whole_word: bool = False,
    overlapping: bool = False,
    normalize_unicode: bool = True,
) -> Dict[str, List[Tuple[int, int]]]:
    
    if normalize_unicode:
        text = unicodedata.normalize("NFC", text)

    # Prepare result with original (possibly unnormalized) term strings as keys
    results: Dict[str, List[Tuple[int, int]]] = {t: [] for t in terms if t}

    # Build a working list of (orig_term, norm_term)
    work_terms: List[Tuple[str, str]] = []
    for t in terms:
        if not t:
            continue
        t_norm = unicodedata.normalize("NFC", t) if normalize_unicode else t
        work_terms.append((t, t_norm))

    flags = 0 if case_sensitive else re.IGNORECASE

    for orig_term, norm_term in work_terms:
        # Escape to match literal text
        escaped = re.escape(norm_term)

        if whole_word:
            # Word boundaries only at the ends (so phrases are supported)
            # Use (?<!\w) and (?!\w) instead of \b to behave well inside lookaheads for overlapping.
            core = f"(?<!\\w){escaped}(?!\\w)"
        else:
            core = escaped

        if overlapping:
            # Lookahead to capture overlapping starts.
            # Put the core inside a capturing group so we can compute the end index.
            pattern = re.compile(f"(?=({core}))", flags)
            matches = pattern.finditer(text)
            for m in matches:
                start = m.start()
                end = start + len(m.group(1))
                results[orig_term].append((start, end))
        else:
            pattern = re.compile(core, flags)
            matches = pattern.finditer(text)
            for m in matches:
                results[orig_term].append((m.start(), m.end()))

    return results


def find_covering_indices(ground_truth: List[Tuple[int, int]], target_spans: List[Tuple[int, int]]) -> List[List[int]]:
    """
    For each target span, return list of indices in ground_truth that are covered/overlapped by it.
    """
    results = []
    for t_start, t_end in target_spans:
        covered_indices = []
        for i, (g_start, g_end) in enumerate(ground_truth):
            # Check if ranges overlap
            if not (g_end < t_start or g_start > t_end):
                covered_indices.append(i)
        results.append(covered_indices)
    return results



class LlaVaProcessor:
    def __init__(self, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = "llava_v1"
        self.model_config = model_config

    def load_demo_images(image_files: Union[List[str], str]):
        if type(image_files) is list:
            out = []
            for image_file in image_files:
                image = Image.open(image_file).convert("RGB")
                out.append(image)
        else:
            out = Image.open(image_files).convert("RGB")
        return out

    # TODO: refactor this, not working
    # def get_processed_tokens_demo(self, text: str, image_files: Union[List[str], str]):
    #     if self.mm_use_im_start_end:
    #         qs = (
    #             qs
    #             + "\n"
    #             + DEFAULT_IM_START_TOKEN
    #             + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    #             + DEFAULT_IM_END_TOKEN
    #             + "\n"
    #             + DEFAULT_IM_START_TOKEN
    #             + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    #             + DEFAULT_IM_END_TOKEN
    #         )
    #     else:
    #         qs = (
    #             qs
    #             + "\n"
    #             + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    #             + "\n"
    #             + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    #         )

    #     conv = conv_templates[self.conv_mode].copy()
    #     conv.append_message(conv.roles[0], text)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     images = self.load_demo_images(image_files)
    #     image_tensor = torch.stack(
    #         [self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
    #     )

    #     input_ids = (
    #         tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    #     )

    #     return image_tensor, input_ids

    def format_text(self, text: str, answer:str=None):
        if self.model_config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        text += answer if answer is not None else ""

        return text

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens(self, text: str, image: Union[str, Image.Image]):
        prompt = self.format_text(text)
        if type(image) is str:
            image = self.load_image(image)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor([image], return_tensors="pt")["pixel_values"]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], images: Union[List[str], List[Image.Image]], batch_answers: List[str], batch_candidates: List[str], batch_hallucination_candidates: List[str]):
        prompt = [self.format_text(text, answer) for text, answer in zip(batch_text, batch_answers)]
        # check if image_paths is a list of images or a list of image paths
        if type(images[0]) is str:
            images = [self.load_image(image_path) for image_path in images]

        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in batch_input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)
        
        ##token level hallucination label
    
        token_level_labels = []
        input_batch = []
        all_answer_masks = []
        for input_ids, candidate_tokens, hal_tokens in zip(batch_input_ids, batch_candidates, batch_hallucination_candidates):
            img_token_pos = torch.where(input_ids==-200)[0]
            special_start_token = torch.where(input_ids==1)[0]
            input_ids_wo_padding = input_ids[input_ids!=0]
            img_token_position_wo_pad = torch.where(input_ids_wo_padding==-200)[0]
            input_ids = torch.cat([input_ids[special_start_token+1:img_token_pos], input_ids[img_token_pos+1:]])
            text = self.tokenizer.decode(input_ids)
            res_start_inx = text.find('ASSISTANT')
            response_text = text[res_start_inx:]
            query_text = text[:res_start_inx]
            
            non_hal_tokens = list(set(candidate_tokens) - set(hal_tokens))
            
            hal_tokens_inx = find_word_spans(response_text.lower(), [i.lower() for i in hal_tokens])
            non_hal_tokens_inx = find_word_spans(response_text.lower(), [i.lower() for i in non_hal_tokens])
            offset = len(query_text)
            hal_tokens_inx = {key: [(start + offset, end + offset-1) for (start, end) in value] for key, value in hal_tokens_inx.items()}
            non_hal_tokens_inx = {key: [(start + offset, end + offset-1) for (start, end) in value] for key, value in non_hal_tokens_inx.items()}
            
            re_tokenized_input_ids = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, padding="max_length", max_length=batch_input_ids.shape[1], return_tensors="pt")
            
            dummy_token_offset_count = (re_tokenized_input_ids["offset_mapping"][0] == torch.tensor([0, 0])).all(dim=1).sum().item()
            token_offsets = [(i[0], i[1]-1) for i in re_tokenized_input_ids["offset_mapping"][0].tolist()[dummy_token_offset_count:]]

            all_mat_tokens = []
            for key, value in hal_tokens_inx.items():
                mat_tokens = find_covering_indices(token_offsets, value)
                mat_tokens = [dummy_token_offset_count + item for sublist in mat_tokens for item in sublist]
                all_mat_tokens.append({"token": key, "mat_tokens": mat_tokens, "label": -1})
            
            for key, value in non_hal_tokens_inx.items():
                mat_tokens = find_covering_indices(token_offsets, value)
                mat_tokens = [dummy_token_offset_count + item for sublist in mat_tokens for item in sublist]
                all_mat_tokens.append({"token": key, "mat_tokens": mat_tokens, "label": 1})
            
            df = pd.DataFrame(all_mat_tokens)
            exploded = df.explode("mat_tokens")
            
            def majority_label(labels):
                counts = Counter(labels)
                return counts.most_common(1)[0][0]

            mapping = (exploded.groupby("mat_tokens")["label"].apply(majority_label).to_dict())
            
            tensor = torch.zeros(batch_input_ids.shape[1], dtype=torch.int)
            for idx, val in mapping.items():
                tensor[idx] = val
            
            
            re_tokenized_input_ids_wo_pad = re_tokenized_input_ids["input_ids"][0][re_tokenized_input_ids["input_ids"][0] !=0]
            adjust_values = input_ids_wo_padding[img_token_position_wo_pad-1:img_token_position_wo_pad+2]
            re_tokenized_input_ids_wo_pad = torch.cat([re_tokenized_input_ids_wo_pad[:img_token_position_wo_pad-1], adjust_values, re_tokenized_input_ids_wo_pad[img_token_position_wo_pad:]])

            pad_len = batch_input_ids.shape[1] - re_tokenized_input_ids_wo_pad.shape[0]
            re_tokenized_input_ids_w_pad = F.pad(re_tokenized_input_ids_wo_pad, (pad_len, 0), mode='constant', value=self.tokenizer.pad_token_id)
        
            # answer mask
            target = torch.tensor([319, 1799, 9047, 13566, 29901])  # token ids for the word "ASSISTANT:"
            candidate_positions = (re_tokenized_input_ids_w_pad == target[0]).nonzero(as_tuple=True)[0]

            start_idx, end_idx = None, None
            for pos in candidate_positions:
                # ensure main slice from pos matches full target
                if torch.equal(re_tokenized_input_ids_w_pad[pos:pos + len(target)], target):
                    start_idx = pos.item()
                    end_idx = pos.item() + len(target) - 1
                    break
            
            ans_mask = torch.arange(re_tokenized_input_ids_w_pad.shape[0])
            ans_mask = (ans_mask >= end_idx+1).to(torch.int)
            all_answer_masks.append(ans_mask)
            
            token_level_labels.append(tensor)
            input_batch.append(re_tokenized_input_ids_w_pad)
        
        batch_input_ids = torch.stack(input_batch)
        token_level_labels = torch.stack(token_level_labels)
        batch_ans_masks = torch.stack(all_answer_masks)

        

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids, token_level_labels, batch_ans_masks
    
    
def collate_fn_builder(processor=None, tokenizer=None):
    def collate_fn(batch):
        bkeys = ["question", "answer", "question_id", "image", "image_id", "image_path", "candidates", "hallucination_candidates"]
        processed_batch = {bkey: [example[bkey] for example in batch] for bkey in bkeys if bkey in batch[0]}

        


        batch_images, batch_text, batch_token_labels, batch_ans_masks = processor.get_processed_tokens_batch(
            [example["question"] for example in batch],
            [example["image_path"] for example in batch],
            [example["answer"] for example in batch],
            [example["candidates"] for example in batch],
            [example["hallucination_candidates"] for example in batch]
        )

        processed_batch["image_tensors"] = batch_images
        processed_batch["input_ids"] = batch_text
        processed_batch["token_level_labels"] = batch_token_labels
        processed_batch["answer_masks"] = batch_ans_masks

        if tokenizer is not None:
            text_inputs = tokenizer(
                [example["question"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]

        return processed_batch

    return collate_fn


def _initialize_dataloader(dataset_name: str, collate_fn, num_workers, batch_size):

        dataset = VQADataset(dataset_name=dataset_name)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        

class VQADataset(Dataset):
    def __init__(self, dataset_name: str):
        
        self.dataset_name = dataset_name
        self.dataset = get_dataset(dataset_name)

    def __repr__(self) -> str:
        return f"{self.dataset_name} {self.split_name} dataset with {len(self)} examples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        content = self.dataset[idx]
        
        data = {
            "image": Image.open(content["image_path"]).convert("RGB"),
            "image_path": content["image_path"],
            "image_id": content["image_id"],
            "question": content["question"],
            "answer": content["answer"],
            "question_id": content["question_id"],
            "candidates": eval(content.get("candidates", '[]')),
            "hallucination_candidates": eval(content.get("hallucination_candidates", '[]')),
        }
        
        return data

    def _get_answer(self, content):
        if self.dataset_name in ["vqa_v2", "okvqa"]:
            return content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
        return content["answer"]
    

 

def get_dataset(dataset_name: str):
    if dataset_name == "pope":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/pope/pope_total_test_9k.csv")
        
        return df.to_dict("records")
    
    elif dataset_name == "chair":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/chair_coco_500/chair_total_test_500.csv")
        return df.to_dict("records")

    elif dataset_name == "amber":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/amber/amber_test_data.csv")
        return df.to_dict("records")
    
    
    elif dataset_name == "holoc_caption":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/haloc/haloc_extension/caption/gemini_labeled_28k.csv")
        return df.to_dict("records")
    