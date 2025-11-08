import argparse
import torch
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
 
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.eval.custom_processor import LlaVaProcessor, collate_fn_builder, _initialize_dataloader
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria

from llava.req_heads.halu_detection import HaluDetectionHead30, HaluDetectionHead24, EvidenceConditionedHallucinationDetector
from llava.req_heads.evidence_head import QueryAdapterMLP, ValueAdapterMLP, SingleHeadQueryAwareScorer, build_importance

from PIL import Image
from uuid import uuid4
import requests
from PIL import Image
from io import BytesIO
import re
import pandas as pd
from tqdm import tqdm
import wandb
import numpy as np

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def llava_forward_halu_detect(batch, tokenizer, model, processor, max_length=128, do_sample=True, num_return_sequences=3):

        conv = conv_templates[processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, input_ids)] if conv.version == "v0" else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()
        ans_masks = batch["answer_masks"].cuda()
        
        attention_mask = (input_ids != tokenizer.pad_token_id).int()
        
        hidden_layers = {}
        def save_hook(layer_id):
            def fn(module, input, output):
                # Detach and move to CPU to avoid GPU memory blowup
                hidden_layers[layer_id] = output[0].detach().cpu()
            return fn
        
        layers_to_hook = [24, 30]
        
        handles = []
        for i, layer in enumerate(model.model.layers):
            if i in layers_to_hook:
                handle = layer.register_forward_hook(save_hook(i))
                handles.append(handle)
        
        with torch.inference_mode():
            output_ids = model.forward(
                input_ids=input_ids,
                attention_mask = attention_mask,
                images=image_tensor.half().cuda(),
                use_cache=False)
        
        for h in handles:
            h.remove()
            

        expanded_input_ids = []
        expanded_ans_masks = []
        for input_id, ans_mask in zip(input_ids, ans_masks):
            
            img_token_position = torch.where(input_id==-200)[0].tolist()[0]
            expanded_input_ids.append(torch.cat((input_id[:img_token_position], torch.full((575,), -200, device=input_id.device), input_id[img_token_position:])))
            expanded_ans_masks.append(torch.cat((ans_mask[:img_token_position], torch.full((575,), 0, device=ans_mask.device), ans_mask[img_token_position:])))
        
        expanded_input_ids = torch.stack(expanded_input_ids).cpu()
        expanded_ans_masks = torch.stack(expanded_ans_masks).cpu()

        target_hl_24_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], expanded_ans_masks)]
        response_ids = [(h[m.bool()]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_input_ids, expanded_ans_masks)]
        
        image_tokens_h1_24_embds = [(h[m == -200]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], expanded_input_ids)]

      
        del (
            input_ids,
            output_ids,
            attention_mask,
            image_tensor,
            ans_masks,
            expanded_input_ids,
            expanded_ans_masks,
            # target_hl_30_embds,
            # target_hl_24_embds,
            # target_labels,
            # target_token_2_bb_masks
            
        )
        torch.cuda.empty_cache()

        return target_hl_24_embds, response_ids, image_tokens_h1_24_embds





from typing import List, Union
import torch

from typing import List, Union
import torch

def words_with_label(tokenizer,
                             input_ids: Union[List[int], torch.Tensor],
                             labels: Union[List[int], torch.Tensor],
                             target_label: int = 2,
                             strict_count_check: bool = False) -> List[str]:
    """
    Group subword tokens back into words and repeat each word as many times
    as the number of its tokens that equal `target_label`.

    - Supports SentencePiece ('▁') and GPT2/BPE ('Ġ') word-start markers.
    - Skips special tokens entirely.
    - If `strict_count_check` is True, asserts that the number of returned
      words equals the total number of tokens with label == target_label.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()

    assert len(input_ids) == len(labels), "input_ids and labels must have the same length"

    toks = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    start_markers = ("▁", "Ġ")

    out_words: List[str] = []

    cur_word_pieces: List[str] = []
    cur_count_target = 0
    in_word = False

    def flush():
        nonlocal cur_word_pieces, cur_count_target, in_word
        if in_word and cur_word_pieces:
            word = "".join(cur_word_pieces)
            if word and cur_count_target > 0:
                out_words.extend([word] * cur_count_target)
        cur_word_pieces = []
        cur_count_target = 0
        in_word = False

    for tid, tok, lab in zip(input_ids, toks, labels):
        # Skip special tokens entirely
        if tid in special_ids or tok is None:
            flush()
            continue

        starts_new = tok.startswith(start_markers)

        if starts_new:
            # finish previous word
            flush()
            in_word = True
            # strip marker and start a new word
            base = tok.lstrip("▁").lstrip("Ġ")
            cur_word_pieces = [base]
            cur_count_target = 1 if lab == target_label else 0
        else:
            # continuation piece
            if not in_word:
                # tokenizer without explicit markers: begin here
                in_word = True
                cur_word_pieces = [tok]
                cur_count_target = 1 if lab == target_label else 0
            else:
                cur_word_pieces.append(tok)
                if lab == target_label:
                    cur_count_target += 1

    # flush the last word
    flush()

    if strict_count_check:
        total_target = sum(int(l == target_label) for l in labels)
        assert total_target == len(out_words), (
            f"Count mismatch: labels have {total_target} occurrences of {target_label}, "
            f"but returned {len(out_words)} words."
        )

    return out_words



def eval_batch_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
    des_weights = torch.load("/Data2/Arun-UAV/NLP/checkpoints/llava_with_des_project_2/mm_projector.bin", map_location='cpu')
    down_scale_des_weights = {}
    for k, v in des_weights.items():
        down_scale_des_weights[k] = v.half()
    model.model.mm_des_projector.load_state_dict(get_w(down_scale_des_weights, 'mm_des_projector'))
    
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
    processor = LlaVaProcessor(tokenizer, image_processor, model.config)
    
    # evidence head loading
    evidence_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/evidence/single_head_strict_train_24l_01_11_2024_d_4096.bin", map_location='cuda')
    single_head_24 = SingleHeadQueryAwareScorer(d = 4096, d_k = 512, mlp_hidden = 512).cuda()
    single_head_24.load_state_dict(evidence_head_24_weights)
    
    # candidate selection head loading
    selection_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/detection/selector_mlp_06_11_2024.bin", map_location='cuda')
    selection_head_24 = HaluDetectionHead24(input_dim= 4096, hidden_dim1 = 2048, hidden_dim2 = 1024).cuda()
    selection_head_24.load_state_dict(selection_head_24_weights)
    
    # detection head loading
    detection_model_type = "attn"
    detection_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/backup_detection/combined_mlp_1000_06_11_2024.bin", map_location='cuda')
    detection_head_24 = HaluDetectionHead24(input_dim= 4096, hidden_dim1 = 2048, hidden_dim2 = 1024).cuda()
    detection_head_24.load_state_dict(detection_head_24_weights)
    

    dataset_name="mme"
    collate_fn = collate_fn_builder(processor, None)
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=32, batch_size=32, shuffle=False)

    
    single_head_24.eval()
    selection_head_24.eval()
    detection_head_24.eval()

    all_dfs = []
    target_columns = ['question', 'answer', 'question_id', 'image_id', 'image_path',  "gt_answer", "data_type","candidates", "hallucination_candidates"]
    for batch in tqdm(dataloader, desc="storing embds"):
        target_hl_24_embds, response_ids, image_tokens_h1_24_embds = llava_forward_halu_detect(batch, tokenizer, model, processor)

        batch_res = []
        for hl_24_embd, response_id, image_tokens_hl_24_embd in zip(target_hl_24_embds, response_ids, image_tokens_h1_24_embds):
            hal_probs = selection_head_24.predict_proba(hl_24_embd)
            hal_pred = (hal_probs >= 0.75).int()
            candidate_words_lbl_1 = words_with_label(tokenizer, input_ids=response_id, labels=hal_pred, target_label=1)
            # candidate_words_lbl_2 = words_with_label(tokenizer, input_ids=response_id, labels=hal_pred, target_label=2)
            # candidate_words_lbl_0 = words_with_label(tokenizer, input_ids=response_id, labels=hal_pred, target_label=0)
            
            # candidate_words_2_hl_24_embd = hl_24_embd[hal_pred == 2]
            # candidate_words_0_hl_24_embd = hl_24_embd[hal_pred == 0]
            candidate_words_1_hl_24_embd = hl_24_embd[hal_pred == 1]
            
            
            res = []
            if candidate_words_1_hl_24_embd.shape[0] != 0:
                assert len(candidate_words_lbl_1) == candidate_words_1_hl_24_embd.shape[0], "mismatch in non-halu words and embeddings"
                with torch.no_grad():
                    evidence_head_logits, _ = single_head_24(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_1_hl_24_embd)
                    evidence_head_probs = torch.sigmoid(evidence_head_logits)
                    if detection_model_type == "attn":
                        detection_head_probs = detection_head_24.predict_proba(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_1_hl_24_embd, evidence_logits=evidence_head_logits)
                    elif detection_model_type == "mlp":
                        detection_head_probs = detection_head_24.predict_proba(x=candidate_words_1_hl_24_embd)

                    for word, evidence, label_prob in zip(candidate_words_lbl_1, evidence_head_probs, detection_head_probs):
                        res.append({"word": word, "evidence": evidence.cpu().numpy(), "label": label_prob.cpu().item()})

            # res = []
            # if candidate_words_2_hl_24_embd.shape[0] != 0:
            #     assert len(candidate_words_lbl_2) == candidate_words_2_hl_24_embd.shape[0], "mismatch in non-halu words and embeddings"
            #     with torch.no_grad():
            #         evidence_head_logits, _ = single_head_24(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_2_hl_24_embd)
            #         evidence_head_probs = torch.sigmoid(evidence_head_logits)
            #         if detection_model_type == "attn":
            #             detection_head_probs = detection_head_24.predict_proba(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_2_hl_24_embd, evidence_logits=evidence_head_logits)
            #         elif detection_model_type == "mlp":
            #             detection_head_probs = detection_head_24.predict_proba(text_tokens=candidate_words_2_hl_24_embd)

            #         for word, evidence, label_prob in zip(candidate_words_lbl_2, evidence_head_probs, detection_head_probs):
            #             res.append({"word": word, "evidence": evidence.cpu().numpy(), "label": label_prob.cpu().item()})

            # if candidate_words_0_hl_24_embd.shape[0] != 0:
            #     assert len(candidate_words_lbl_0) == candidate_words_0_hl_24_embd.shape[0], "mismatch in halu words and embeddings"
            #     with torch.no_grad():
            #         evidence_head_logits, _ = single_head_24(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_0_hl_24_embd)
            #         evidence_head_probs = torch.sigmoid(evidence_head_logits)
            #         if detection_model_type == "attn":
            #             detection_head_probs = detection_head_24.predict_proba(img_tokens=image_tokens_hl_24_embd, text_tokens=candidate_words_0_hl_24_embd, evidence_logits=evidence_head_logits)
            #         elif detection_model_type == "mlp":
            #             detection_head_probs = detection_head_24.predict_proba(text_tokens=candidate_words_0_hl_24_embd)
                        
            #         for word, evidence, label_prob in zip(candidate_words_lbl_0, evidence_head_probs, detection_head_probs):
            #             res.append({"word": word, "evidence": evidence.cpu().numpy(), "label": label_prob.cpu().item()})

            batch_res.append(res)

        
        target_values = {i:batch[i]  for i in target_columns}
        df = pd.DataFrame(target_values)
        df["labels_with_evidence"] = batch_res
        all_dfs.append(df)

    total_df = pd.concat(all_dfs)
    arc = "attn"
    date = "09_11_2025"
    train_type= "combined_stage"
    if dataset_name == "chair":
        total_df.to_pickle(f"/Data2/Arun-UAV/NLP/vision_halu/total_flow_testing_results/chair/{train_type}_label_with_evidence_and_{arc}_{date}.pkl")

    elif dataset_name == "pope":
        total_df.to_pickle(f"/Data2/Arun-UAV/NLP/vision_halu/total_flow_testing_results/pope/{train_type}_label_with_evidence_and_{arc}_{date}.pkl")

    elif dataset_name == "mme":
        total_df.to_pickle(f"/Data2/Arun-UAV/NLP/vision_halu/total_flow_testing_results/mme/{train_type}_label_with_evidence_and_{arc}_{date}.pkl")

    elif dataset_name == "amber":
        total_df.to_pickle(f"/Data2/Arun-UAV/NLP/vision_halu/total_flow_testing_results/amber/{train_type}_label_with_evidence_and_{arc}_{date}.pkl")

    elif dataset_name == "ours":
        total_df.to_pickle(f"/Data2/Arun-UAV/NLP/vision_halu/total_flow_testing_results/ours/{train_type}_label_with_evidence_and_{arc}_{date}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=25)
    args = parser.parse_args()

    eval_batch_model(args)