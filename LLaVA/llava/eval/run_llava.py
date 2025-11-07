import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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

from llava.req_heads.halu_detection import SingleHeadDetectionClassifier, HaluDetectionHead24, EvidenceConditionedHallucinationDetector
from llava.req_heads.evidence_head import SingleHeadQueryAwareScorer

from PIL import Image
from uuid import uuid4
import requests
from PIL import Image
from io import BytesIO
import re
import pandas as pd
from tqdm import tqdm
import wandb

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
        for input_id, token_level_labels, ans_mask in zip(input_ids, token_level_labels, ans_masks):
            
            img_token_position = torch.where(input_id==-200)[0].tolist()[0]
            expanded_input_ids.append(torch.cat((input_id[:img_token_position], torch.full((575,), -200, device=input_id.device), input_id[img_token_position:])))
            expanded_token_level_labels.append(torch.cat((token_level_labels[:img_token_position], torch.full((575,), 0, device=token_level_labels.device), token_level_labels[img_token_position:])))
            expanded_ans_masks.append(torch.cat((ans_mask[:img_token_position], torch.full((575,), 0, device=ans_mask.device), ans_mask[img_token_position:])))
        
        expanded_input_ids = torch.stack(expanded_input_ids).cpu()
        expanded_token_level_labels = torch.stack(expanded_token_level_labels).cpu()
        expanded_ans_masks = torch.stack(expanded_ans_masks).cpu()
        
        ans_only_token_level_labels =  torch.stack([h*m for h, m in zip(expanded_token_level_labels, expanded_ans_masks)])
        

        target_hl_30_embds = [(h[m != 0]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], ans_only_token_level_labels)]
        target_hl_24_embds = [(h[m != 0]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], ans_only_token_level_labels)]
        target_labels = [(h[m != 0]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_token_level_labels, ans_only_token_level_labels)]
        response_ids = [h[m != 0].detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_input_ids, ans_only_token_level_labels)]

        
        # target_hl_30_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[30], expanded_ans_masks)]
        # target_hl_24_embds = [(h[m.bool()]).detach().clone().float().cuda().requires_grad_(False) for h, m in zip(hidden_layers[24], expanded_ans_masks)]
        # target_labels = [(h[m.bool()]).detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_token_level_labels, expanded_ans_masks)]
        # response_ids = [h[m.bool()].detach().clone().long().cuda().requires_grad_(False) for h, m in zip(expanded_input_ids, expanded_ans_masks)]

        
        
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
        )
        torch.cuda.empty_cache()
            
        

        return target_hl_30_embds, target_hl_24_embds, target_labels, image_tokens_h1_30_embds, image_tokens_h1_24_embds


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
    
    #evidence head loading
    # evidence_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/evidence/single_head_strict_train_24l_01_11_2024_d_4096.bin", map_location='cuda')
    # evidence_head_24 = SingleHeadQueryAwareScorer(d = 4096, d_k = 512, mlp_hidden = 512).cuda()
    # evidence_head_24.load_state_dict(evidence_head_24_weights)


    dataset_name="holoc_total_train_gemini_labels"
    collate_fn = collate_fn_builder(processor, None)
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=64, batch_size=64, shuffle=True)

    
    
    # detection_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/detection/short_train_attn_detection_head_24hl_05_11_2024.bin", map_location='cuda')
    # detection_head_24 = SingleHeadDetectionClassifier(d = 4096, d_k = 1024, mlp_hidden = 1024).cuda()
    # detection_head_24.load_state_dict(detection_head_24_weights)
    
    # detection_head_24_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/detection/short_mlp_06_11_2024.bin", map_location='cuda')
    detection_head_24 = HaluDetectionHead24(input_dim= 4096, hidden_dim1 = 2048, hidden_dim2 = 1024).cuda()
    # detection_head_24.load_state_dict(detection_head_24_weights)
    
    # detection_head_24 = EvidenceConditionedHallucinationDetector(d= 4096).cuda()
    
    
    
    optimizer_detection_head_24 = AdamW(detection_head_24.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    
    steps_per_epoch = len(dataloader)
    num_epochs = 1                             # <- set this
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)     # 10% warmup
    min_lr_ratio = 0.05
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay from 1.0 -> min_lr_ratio
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    
    with torch.no_grad():
        print("just init 24:", any(torch.isnan(p).any() for p in detection_head_24.parameters()))
    
    sched_optimizer_detection_head_24 = LambdaLR(optimizer_detection_head_24, lr_lambda)

    step = 0
    detection_head_24.train()
    for batch in tqdm(dataloader, desc=f"step: training detection head"):
        target_hl_30_embds, target_hl_24_embds, target_labels, image_tokens_h1_30_embds, image_tokens_h1_24_embds = generate_llava(batch, tokenizer, model, processor)
        
        optimizer_detection_head_24.zero_grad(set_to_none=True)
        
        losses_24 = []
        for hl_30_embd, hl_24_embd, target_label, img_token_h1_30_embd, img_token_h1_24_embd in \
                zip(target_hl_30_embds, target_hl_24_embds, target_labels, image_tokens_h1_30_embds, image_tokens_h1_24_embds):
            
            # with torch.no_grad():
            #     ev_logits, _= evidence_head_24(img_tokens=img_token_h1_24_embd, text_tokens=hl_24_embd)
            
            labels_mapped = (target_label == 1).float()
            # logits, loss = detection_head_24(img_tokens=img_token_h1_24_embd, text_tokens=hl_24_embd, labels=labels_mapped, evidence_logits=ev_logits)
            # logits, loss = detection_head_24(img_tokens=img_token_h1_24_embd, text_tokens=hl_24_embd, labels=labels_mapped)
            logits, loss = detection_head_24(x=hl_24_embd, labels=labels_mapped)
            losses_24.append(loss)
            
        batch_loss_24 = torch.stack(losses_24).mean()
        
        if torch.isnan(batch_loss_24):
            print(f"NaN loss at step {step}, skipping batch")
            continue
        
        batch_loss_24.backward()

        torch.nn.utils.clip_grad_norm_(list(detection_head_24.parameters()), max_norm=1.0)

        optimizer_detection_head_24.step()
        sched_optimizer_detection_head_24.step()

        print(f"24 Avg loss: {batch_loss_24.item():.4f}")
        wandb.log({
            "step": step,
            "avg_loss_24": batch_loss_24.item(),
            "lr_24": optimizer_detection_head_24.param_groups[0]["lr"]
        })
        step += 1
        if step in [500,1000,1500,2000, 2250, 2500]:
            torch.save(detection_head_24.state_dict(), f"/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/backup_detection/combined_mlp_{step}_06_11_2024.bin")

    torch.save(detection_head_24.state_dict(), "/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/detection/combined_mlp_06_11_2024.bin")

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

    detection_head_30_weights = torch.load("/Data2/Arun-UAV/NLP/vision_halu/head_checkpoints/detection/total_train_head_24l_26_20_2024_d_8192.bin", map_location='cuda')
    detection_head_30 = HaluDetectionHead24().cuda()
    detection_head_30.load_state_dict(detection_head_30_weights)

    dataset_name="chair_caption_gen"
    collate_fn = collate_fn_builder(processor, None)
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=16, batch_size=16, shuffle=False)
    
    detection_head_30.eval()
    all_dfs = []
    for batch in tqdm(dataloader, desc="storing embds"):
        class_mapping  = {-1:0, 0:1, 1:2}
        hl_30_embds, hl_24_embds, target_labels, response_ids = generate_llava(batch, tokenizer, model, processor, mode="eval")
        
        all_res = []
        for hl_30_embd, target_label, response_id in zip(hl_24_embds, target_labels, response_ids):

            labels_mapped = torch.where(target_label == -1, 0, torch.where(target_label == 0, 1, 2))
            logits_30 = detection_head_30(hl_30_embd)
            pred_class_30 = torch.argmax(logits_30, dim=1)
            all_res.append((tokenizer.decode(response_id), 
                            tokenizer.decode(response_id[torch.where(labels_mapped == 0)]).split(" "), 
                            tokenizer.decode(response_id[torch.where(pred_class_30 == 0)]).split(" ")))

        df = pd.DataFrame(all_res, columns=["full_response", "true_hallu_tokens", "pred_hallu_tokens"])
        df["image_path"] = batch["image_path"]
        all_dfs.append(df)

    total_df = pd.concat(all_dfs)
    total_df.to_csv("/Data2/Arun-UAV/NLP/vision_halu/testing_res/halu_detection/chair_res_24l_26_20_2024_d_8192.csv", index=False) 



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

    # eval_batch_model(args)
    train_batch_model(args)