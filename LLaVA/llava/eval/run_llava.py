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

from llava.req_heads.halu_detection import HaluDetectionHead30, HaluDetectionHead24
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

def model_infer_4_generation(batch, tokenizer, model, processor, max_length=128, do_sample=True, num_return_sequences=3):

        conv = conv_templates[processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, input_ids)] if conv.version == "v0" else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        img_token_imp_scores = torch.tensor(np.stack(batch["img_token_imp_scores"]))
        
        input_ids = input_ids.cuda()

        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            img_token_weights =img_token_imp_scores
        )
        generated_outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_outputs = [out.strip() for out in generated_outputs]
        generated_outputs = [out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs]

        return generated_outputs

def benchmark_batch_eval(args):
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
    

    dataset_name="chair"
    collate_fn = collate_fn_builder(processor, None, mode="gen")
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=32, batch_size=32)

    all_generated_captions = []
    question_ids = []
    questions = []
    answers = []
    all_dfs = []
    for batch in tqdm(dataloader, desc="Generating captions"):
        generated_captions = model_infer_4_generation(batch, tokenizer, model, processor)
        df = pd.DataFrame({k:v for k,v in batch.items() if k not in ['image_tensors', 'input_ids', 'image']})
        df["generated_captions"] = generated_captions
        all_dfs.append(df)
        # all_generated_captions.extend(generated_captions)
        # question_ids.extend(batch["question_id"])
        # questions.extend(batch["answer"])
    
    result_df = pd.concat(all_dfs)
    if dataset_name == "chair":
        result_df.to_json("/Data2/Arun-UAV/NLP/vision_halu/evidence_head_test_datasets/chair/chair_llava_des_imp_itr_1_02_11_2025.jsonl", lines=True, orient="records")
    elif dataset_name == "pope":
        result_df.to_json("/Data2/Arun-UAV/NLP/vision_halu/testing_res/pope_llava_base_des_02_11_2025.jsonl", lines=True, orient="records")
    elif dataset_name == "amber":
        req_df = result_df[["question_id", "generated_captions"]]
        req_df.columns = ["id", "response"]
        req_df.to_json("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/amber/llava_res/amber_llava_base_des_02_11_2025.json", lines=True, orient="records")
        result_df.to_json("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/amber/llava_res/amber_llava_base_des_all_res_02_11_2025.json", lines=True, orient="records")
    
    



def generate_llava(batch, tokenizer, model, processor, mode = "train", max_length=128, do_sample=True, num_return_sequences=3):

        conv = conv_templates[processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, input_ids)] if conv.version == "v0" else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()
        token_level_labels = batch["token_level_labels"].cuda()
        ans_masks = batch["answer_masks"].cuda()
        token_2_bb_masks = batch["token_2_bb_masks"].cuda()
        
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
        expanded_token_level_labels = []
        expanded_ans_masks = []
        expanded_token_2_bb_masks = []
        img_token_position = []
        for input_id, token_level_labels, ans_mask, token_2_bb_mask in zip(input_ids, token_level_labels, ans_masks, token_2_bb_masks):
            
            img_token_position = torch.where(input_id==-200)[0].tolist()[0]
            expanded_input_ids.append(torch.cat((input_id[:img_token_position], torch.full((575,), -200, device=input_id.device), input_id[img_token_position:])))
            expanded_token_level_labels.append(torch.cat((token_level_labels[:img_token_position], torch.full((575,), 0, device=token_level_labels.device), token_level_labels[img_token_position:])))
            expanded_ans_masks.append(torch.cat((ans_mask[:img_token_position], torch.full((575,), 0, device=ans_mask.device), ans_mask[img_token_position:])))
            expanded_token_2_bb_masks.append(torch.cat((token_2_bb_mask[:img_token_position], torch.full((575, 576), 0, device=token_2_bb_mask.device), token_2_bb_mask[img_token_position:])))
        
        expanded_input_ids = torch.stack(expanded_input_ids).cpu()
        expanded_token_level_labels = torch.stack(expanded_token_level_labels).cpu()
        expanded_ans_masks = torch.stack(expanded_ans_masks).cpu()
        expanded_token_2_bb_masks = torch.stack(expanded_token_2_bb_masks).cpu()

        ans_only_token_level_labels =  torch.stack([h*m for h, m in zip(expanded_token_level_labels, expanded_ans_masks)])

        target_hl_30_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], ans_only_token_level_labels)]
        target_hl_24_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], ans_only_token_level_labels)]
        target_token_2_bb_masks = [(h[m.bool()]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_token_2_bb_masks, ans_only_token_level_labels)]
        response_ids = [(h[m.bool()]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_input_ids, ans_only_token_level_labels)]

        image_tokens_h1_30_embds = [ (h[m == -200]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], expanded_input_ids)]
        image_tokens_h1_24_embds = [ (h[m == -200]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], expanded_input_ids)]

        if mode == "train":
            del (
                input_ids,
                output_ids,
                attention_mask,
                image_tensor,
                ans_masks,
                expanded_input_ids,
                expanded_token_level_labels,
                expanded_ans_masks,
                expanded_token_2_bb_masks,
                ans_only_token_level_labels,
                # target_hl_30_embds,
                # target_hl_24_embds,
                # target_labels,
                # target_token_2_bb_masks
                
            )
            torch.cuda.empty_cache()

            return target_hl_30_embds, target_hl_24_embds, target_token_2_bb_masks, image_tokens_h1_30_embds, image_tokens_h1_24_embds

def generate_llava_eval(batch, tokenizer, model, processor, max_length=128, do_sample=True, num_return_sequences=3):

        conv = conv_templates[processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, input_ids)] if conv.version == "v0" else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()
        token_level_labels = batch["token_level_labels"].cuda()
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
        expanded_token_level_labels = []
        expanded_ans_masks = []
        img_token_position = []
        for input_id, token_level_labels, ans_mask in zip(input_ids, token_level_labels, ans_masks):
            
            img_token_position = torch.where(input_id==-200)[0].tolist()[0]
            expanded_input_ids.append(torch.cat((input_id[:img_token_position], torch.full((575,), -200, device=input_id.device), input_id[img_token_position:])))
            expanded_token_level_labels.append(torch.cat((token_level_labels[:img_token_position], torch.full((575,), 0, device=token_level_labels.device), token_level_labels[img_token_position:])))
            expanded_ans_masks.append(torch.cat((ans_mask[:img_token_position], torch.full((575,), 0, device=ans_mask.device), ans_mask[img_token_position:])))
        
        expanded_input_ids = torch.stack(expanded_input_ids).cpu()
        expanded_token_level_labels = torch.stack(expanded_token_level_labels).cpu()
        expanded_ans_masks = torch.stack(expanded_ans_masks).cpu()

        ans_only_token_level_labels =  torch.stack([h*m for h, m in zip(expanded_token_level_labels, expanded_ans_masks)])

        target_hl_30_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], ans_only_token_level_labels)]
        target_hl_24_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], ans_only_token_level_labels)]
        response_ids = [(h[m.bool()]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_input_ids, ans_only_token_level_labels)]

        image_tokens_h1_30_embds = [ (h[m == -200]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], expanded_input_ids)]
        image_tokens_h1_24_embds = [ (h[m == -200]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], expanded_input_ids)]

      
        del (
            input_ids,
            output_ids,
            attention_mask,
            image_tensor,
            ans_masks,
            expanded_input_ids,
            expanded_token_level_labels,
            expanded_ans_masks,
            ans_only_token_level_labels,
            # target_hl_30_embds,
            # target_hl_24_embds,
            # target_labels,
            # target_token_2_bb_masks
            
        )
        torch.cuda.empty_cache()

        return target_hl_30_embds, target_hl_24_embds, image_tokens_h1_30_embds, image_tokens_h1_24_embds




def train_batch_model(args):
    
    wandb.init(
        project="mlp-3class-classifier",
        config={
            "input_dim": 4096,
            "hidden_dim": 1024,
            "dropout": 0.3,
            "num_classes": 3,
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 1e-3,
            "optimizer": "Adam"
        }
    )
    config = wandb.config

    # disable_torch_init()
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
    

    dataset_name="total_evidence_head_train"
    collate_fn = collate_fn_builder(processor, None)
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=64, batch_size=64, shuffle=True)

    # value_head_24 = ValueAdapterMLP(d = 4096, hidden=1024, d_k = 512).cuda()
    # query_head_24 = QueryAdapterMLP(d = 4096, hidden=1024, d_k = 512).cuda()

    single_head_24 = SingleHeadQueryAwareScorer( d = 4096, d_k = 512, mlp_hidden = 512).cuda()


    # with torch.no_grad():
    #     print("just init 30:", any(torch.isnan(p).any() for p in value_head_24.parameters()))
    #     print("just init 24:", any(torch.isnan(p).any() for p in query_head_24.parameters()))
    with torch.no_grad():
        print("just init 24:", any(torch.isnan(p).any() for p in single_head_24.parameters()))

    
        
    # optimizer_value = AdamW(value_head_24.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    # optimizer_query = AdamW(query_head_24.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))

    optimizer_single_head_24 = AdamW(single_head_24.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))

    # 2) Scheduler setup
    steps_per_epoch = len(dataloader)
    num_epochs = 1                             # <- set this
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)     # 10% warmup
    min_lr_ratio = 0.05                        # final LR will be 5% of base

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay from 1.0 -> min_lr_ratio
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    # sched_value = LambdaLR(optimizer_value, lr_lambda)
    # sched_query = LambdaLR(optimizer_query, lr_lambda)
    
    sched_single_head_24 = LambdaLR(optimizer_single_head_24, lr_lambda)

    step = 0
    for epoch in range(num_epochs):
        # value_head_24.train(); query_head_24.train()
        single_head_24.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # optimizer_value.zero_grad(set_to_none=True)
            # optimizer_query.zero_grad(set_to_none=True)
            optimizer_single_head_24.zero_grad(set_to_none=True)

            losses_24 = []
            target_hl_30_embds, target_hl_24_embds, target_token_2_bb_masks, image_tokens_h1_30_embds, image_tokens_h1_24_embds = \
                generate_llava(batch, tokenizer, model, processor)

            for hl_30_embd, hl_24_embd, token_2_bb_mask, img_token_h1_30_embd, img_token_h1_24_embd in \
                zip(target_hl_30_embds, target_hl_24_embds, target_token_2_bb_masks, image_tokens_h1_30_embds, image_tokens_h1_24_embds):
                # q = query_head_24(hl_24_embd)
                # v = value_head_24(img_token_h1_24_embd)
                loss  = single_head_24(img_tokens=img_token_h1_24_embd, text_tokens=hl_24_embd, labels=token_2_bb_mask)
                losses_24.append(loss)
                
                

            batch_loss_24 = torch.stack(losses_24).mean()
            batch_loss_24.backward()

            # (optional) gradient clipping helps with stability
            # torch.nn.utils.clip_grad_norm_(list(query_head_24.parameters()) + list(value_head_24.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(list(single_head_24.parameters()), max_norm=1.0)

            
            # optimizer_value.step()
            # optimizer_query.step()
            optimizer_single_head_24.step()

            # Step schedulers *after* optimizer.step() for LambdaLR
            # sched_value.step()
            # sched_query.step()
            sched_single_head_24.step()
            
            
            print(f"24 Avg loss: {batch_loss_24.item():.4f}")
            wandb.log({
                "step": step,
                "batch_loss_24": batch_loss_24,
                "lr_24": optimizer_single_head_24.param_groups[0]["lr"]
            })
            step += 1
            if step in [50, 500, 1000, 1500, 2000]:
                 torch.save(single_head_24.state_dict(), f"/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/evidence/single_head_strict_train_24l_01_11_2024_d_4096_step{step}.bin")
    
    torch.save(single_head_24.state_dict(), "/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/evidence/single_head_strict_train_24l_01_11_2024_d_4096.bin")

    torch.cuda.empty_cache()


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

    evidence_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/evidence/single_head_strict_train_24l_01_11_2024_d_4096.bin", map_location='cuda')
    single_head_24 = SingleHeadQueryAwareScorer( d = 4096, d_k = 512, mlp_hidden = 512).cuda()
    single_head_24.load_state_dict(evidence_head_24_weights)

    dataset_name="chair"
    collate_fn = collate_fn_builder(processor, None, mode = "eval")
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=64, batch_size=64, shuffle=False)
    
    single_head_24.eval()
    all_dfs = []
    for batch in tqdm(dataloader, desc="storing embds"):
        target_hl_30_embds, target_hl_24_embds, image_tokens_h1_30_embds, image_tokens_h1_24_embds = generate_llava_eval(batch, tokenizer, model, processor)

        target_columns = ['question', 'answer', 'question_id', 'image_id', 'image_path', 'candidates']
        all_seq_img_imp_scores = []
        for hl_30_embd, hl_24_embd, img_token_h1_30_embd, img_token_h1_24_embd, candidates in zip(target_hl_30_embds, target_hl_24_embds, image_tokens_h1_30_embds, image_tokens_h1_24_embds, batch["candidates"]):
            single_pred = single_head_24.predict_proba(img_tokens = img_token_h1_24_embd, text_tokens = hl_24_embd)
            img_token_imp_score = build_importance(single_pred)
            all_seq_img_imp_scores.append(img_token_imp_score[0].cpu().numpy())
        
        target_values = {i:batch[i]  for i in target_columns}
        df = pd.DataFrame(target_values)
        df["img_token_imp_scores"] = all_seq_img_imp_scores
        all_dfs.append(df)

    total_df = pd.concat(all_dfs)
    total_df.to_pickle("/Data2/Arun-UAV/NLP/vision_halu/evidence_head_test_datasets/chair/base_des_imp_scores_01_11_2025.pkl")



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
    # benchmark_batch_eval(args)
    # train_batch_model(args)