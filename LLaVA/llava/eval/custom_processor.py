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
import math
from typing import List, Dict, Tuple, Union
import torch 

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


BBox = Union[Dict[str, float], Dict[str, int], Tuple[float, float, float, float], List[float]]

def _coerce_bbox(b: BBox) -> Tuple[float, float, float, float]:
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if isinstance(b, dict):
        keys = {k.lower(): k for k in b.keys()}
        def g(k):
            for cand in (k, k.replace('main', 'min')):  # tolerate 'ymain' -> 'ymin'
                if cand in keys:
                    return float(b[keys[cand]])
            raise KeyError(f"Missing key {k} in bbox dict: {b}")
        return g('xmin'), g('ymin'), g('xmax'), g('ymax')
    raise TypeError(f"Unsupported bbox format: {type(b)}")

def bbox_patch_binary_masks(
    image_w: int,
    image_h: int,
    bboxes: Union[BBox, List[BBox]],
    patch_w: int = 14,
    patch_h: int = None,
    *,
    inclusive_xymax: bool = False
) -> List[List[int]]:
    """
    For a W×H image split into patch_w×patch_h patches (row-major),
    return, for each bbox, a flat 0/1 list of length (num_rows*num_cols)
    marking patches that intersect the bbox.

    Args:
        image_w, image_h: image size in pixels (e.g., 336, 336)
        bboxes: single bbox or list of bboxes. Each bbox can be:
                - dict with keys xmin,ymin,xmax,ymax (case-insensitive; 'ymain' tolerated)
                - list/tuple [xmin, ymin, xmax, ymax]
        patch_w, patch_h: patch size in pixels (default 14×14). If patch_h is None, uses patch_w.
        inclusive_xymax: treat (xmax,ymax) as inclusive if True; exclusive if False.

    Returns:
        List of binary masks (one per bbox). Each mask is a list[int] of length num_rows*num_cols.
        Indexing is row-major: idx = row * num_cols + col.
    """
    if patch_h is None:
        patch_h = patch_w

    num_cols = math.ceil(image_w / patch_w)
    num_rows = math.ceil(image_h / patch_h)
    total_patches = num_rows * num_cols

    # Normalize to list of bboxes
    if isinstance(bboxes, list) and not (len(bboxes) == 4 and all(isinstance(v, (int, float)) for v in bboxes)):
        bbox_list = bboxes
    else:
        bbox_list = [bboxes]

    masks: List[List[int]] = []

    for b in bbox_list:
        x0, y0, x1, y1 = _coerce_bbox(b)
        if inclusive_xymax:
            x1 += 1.0
            y1 += 1.0

        # Clamp to image bounds
        x0 = max(0.0, min(x0, float(image_w)))
        y0 = max(0.0, min(y0, float(image_h)))
        x1 = max(0.0, min(x1, float(image_w)))
        y1 = max(0.0, min(y1, float(image_h)))

        # Degenerate -> all zeros
        if x1 <= x0 or y1 <= y0:
            masks.append([0] * total_patches)
            continue

        # Candidate patch span
        col_start = int(math.floor(x0 / patch_w))
        col_end   = int(math.ceil (x1 / patch_w) - 1)
        row_start = int(math.floor(y0 / patch_h))
        row_end   = int(math.ceil (y1 / patch_h) - 1)

        col_start = max(0, min(col_start, num_cols - 1))
        col_end   = max(0, min(col_end,   num_cols - 1))
        row_start = max(0, min(row_start, num_rows - 1))
        row_end   = max(0, min(row_end,   num_rows - 1))

        mask = [0] * total_patches

        # Mark intersecting patches
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                patch_x0 = c * patch_w
                patch_y0 = r * patch_h
                patch_x1 = min(patch_x0 + patch_w, image_w)
                patch_y1 = min(patch_y0 + patch_h, image_h)

                inter_w = max(0.0, min(x1, patch_x1) - max(x0, patch_x0))
                inter_h = max(0.0, min(y1, patch_y1) - max(y0, patch_y0))
                if inter_w > 0.0 and inter_h > 0.0:
                    idx = r * num_cols + c
                    mask[idx] = 1

        masks.append(mask)

    return masks



def combine_mask_tensor(masks: List[List[int]], *, out_dtype=torch.int) -> torch.Tensor:
    if len(masks) == 0:
        return torch.empty((0,), dtype=out_dtype)
    mask_tensor = torch.as_tensor(masks, dtype=out_dtype)
    
    if mask_tensor.numel() == 0:
        # Empty input -> empty output
        return torch.empty((mask_tensor.shape[-1] if mask_tensor.ndim == 2 else 0,), 
                           dtype=out_dtype, device=mask_tensor.device)

    # Normalize to bool
    mask_bool = (mask_tensor != 0) if mask_tensor.dtype != torch.bool else mask_tensor
    # Row-wise OR -> single row
    combined_bool = torch.any(mask_bool, dim=0)
    return combined_bool.to(out_dtype)




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
        all_token_2_bb_masks = []
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
            
            candidates_bb_info = candidate_tokens
            non_hal_tokens = [i["word"] for i in candidates_bb_info]
            
            hal_tokens_inx = find_word_spans(response_text.lower(), [i.lower() for i in hal_tokens])
            non_hal_tokens_inx = find_word_spans(response_text.lower(), [i.lower() for i in non_hal_tokens])
            offset = len(query_text)
            hal_tokens_inx = {key: [(start + offset, end + offset-1) for (start, end) in value] for key, value in hal_tokens_inx.items()}
            non_hal_tokens_inx = {key: [(start + offset, end + offset-1) for (start, end) in value] for key, value in non_hal_tokens_inx.items()}
            
            re_tokenized_input_ids = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, padding="max_length", max_length=batch_input_ids.shape[1], return_tensors="pt")
            
            dummy_token_offset_count = (re_tokenized_input_ids["offset_mapping"][0] == torch.tensor([0, 0])).all(dim=1).sum().item()
            token_offsets = [(i[0], i[1]-1) for i in re_tokenized_input_ids["offset_mapping"][0].tolist()[dummy_token_offset_count:]]

            all_mat_tokens = []
            assert len(hal_tokens_inx) == 0, "hallucination tokens must be empty"
            for key, value in hal_tokens_inx.items():
                mat_tokens = find_covering_indices(token_offsets, value)
                mat_tokens = [dummy_token_offset_count + item for sublist in mat_tokens for item in sublist]
                all_mat_tokens.append({"token": key, "mat_tokens": mat_tokens, "label": -1})
            
            for key, value in non_hal_tokens_inx.items():
                mat_tokens_full = find_covering_indices(token_offsets, value)
                mat_tokens = [dummy_token_offset_count + sublist[-1] for sublist in mat_tokens_full if sublist]
                all_mat_tokens.append({"token": key, "mat_tokens": mat_tokens, "label": 1})
            
            df = pd.DataFrame(all_mat_tokens)
            bb_df = pd.DataFrame(candidates_bb_info)
            bb_df = bb_df.rename(columns= {"word": "token"})
            final_df = pd.merge(df, bb_df, on="token", how="left")
            final_df = final_df.dropna(subset=["mat_tokens", "bbox"])
            final_df = final_df[final_df["mat_tokens"].apply(lambda x: len(x)>0)]
            final_df = final_df[final_df["bbox"].apply(lambda x: len(x)>0)]
            exploded = final_df.explode("mat_tokens")
            
            
            def majority_label(labels):
                counts = Counter(labels)
                return counts.most_common(1)[0][0]

            mapping = (exploded.groupby("mat_tokens")["label"].apply(majority_label).to_dict())
            
            mapping = {}
            bb_token_mapping = {}
            for inx, row in exploded.iterrows():
                mapping[row["mat_tokens"]] = row["label"]
                bb_token_mapping[row["mat_tokens"]] = row["bbox"]
                
            tensor = torch.zeros(batch_input_ids.shape[1], dtype=torch.int)
            for idx, val in mapping.items():
                tensor[idx] = val
            
            exp_token_2_bb = torch.zeros((batch_input_ids.shape[1], 576), dtype=torch.int)
            try:
                for idx, val in bb_token_mapping.items():
                    bb_mask =  bbox_patch_binary_masks(image_w=336, image_h=336, bboxes=val, patch_w=14, patch_h=14)
                    flattern_bb_mask = combine_mask_tensor(bb_mask, out_dtype=torch.int)
                    exp_token_2_bb[idx] = flattern_bb_mask
            except Exception as e:
                print(e)
                print("error in bb mask creation")

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
            all_token_2_bb_masks.append(exp_token_2_bb)
        
        batch_input_ids = torch.stack(input_batch)
        token_level_labels = torch.stack(token_level_labels)
        batch_ans_masks = torch.stack(all_answer_masks)
        batch_token_2_bb_masks = torch.stack(all_token_2_bb_masks)

        

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids, token_level_labels, batch_ans_masks, batch_token_2_bb_masks
    
    
def collate_fn_builder(processor=None, tokenizer=None):
    def collate_fn(batch):
        bkeys = ["question", "answer", "question_id", "image", "image_id", "image_path", "candidates", "hallucination_candidates"]
        processed_batch = {bkey: [example[bkey] for example in batch] for bkey in bkeys if bkey in batch[0]}

        


        batch_images, batch_text, batch_token_labels, batch_ans_masks, batch_token_2_bb_masks = processor.get_processed_tokens_batch(
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
        processed_batch["token_2_bb_masks"] = batch_token_2_bb_masks

        if tokenizer is not None:
            text_inputs = tokenizer(
                [example["question"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]

        return processed_batch

    return collate_fn


def _initialize_dataloader(dataset_name: str, collate_fn, num_workers, batch_size, shuffle = False):

        dataset = VQADataset(dataset_name=dataset_name)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
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
    
    
    elif dataset_name == "holoc_total_train_gemini_labels":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/haloc/haloc_extension/caption/gemini_labeled_28k.csv")
        df_1 = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/haloc/haloc_extension/instruct/gemini_labeled_40k.csv")
        df_2 = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/haloc/haloc_extension/vqa/tp_data.csv")
        df_3 = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/haloc/haloc_extension/vqa/tn_data.csv")
        total_df = pd.concat([df, df_1, df_2, df_3])
        total_df = total_df.sample(frac=1)
        return total_df.to_dict("records")

    elif dataset_name == "coco_evidence_head_train":
        df = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/evidence_head_train_datasets/coco_long_captions/downscaled_total_coco_evidence_head_train_data_15k.csv")
        return df.to_dict("records")
