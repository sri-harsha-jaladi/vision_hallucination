import argparse
import torch

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

from PIL import Image
from uuid import uuid4
import requests
from PIL import Image
from io import BytesIO
import re
import pandas as pd
from tqdm import tqdm

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




def generate_llava(batch, tokenizer, model, processor, max_length=128, do_sample=True, num_return_sequences=3):

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
        
        

        # output_ids = model.generate(
        #     input_ids,
        #     attention_mask = attention_mask,
        #     images=image_tensor.half().cuda(),
        #     do_sample=True if args.temperature > 0 else False,
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     num_beams=args.num_beams,
        #     max_new_tokens=args.max_new_tokens,
        #     use_cache=True,
        #     stopping_criteria=stopping_criteria,
        #     # num_return_sequences=num_return_sequences,
        # )
        # generated_outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # generated_outputs = [out.strip() for out in generated_outputs]
        # generated_outputs = [out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs]
        
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

        target_hl_30_embds = [h[m.bool()] for h, m in zip(hidden_layers[30], expanded_ans_masks)]
        target_hl_24_embds = [h[m.bool()] for h, m in zip(hidden_layers[24], expanded_ans_masks)]
        target_labels = [h[m.bool()] for h, m in zip(expanded_token_level_labels, expanded_ans_masks)]
        df_24 = pd.DataFrame({"embds": [j for i in target_hl_24_embds for j in i.tolist()], "labels": [j for i in target_labels for j in i.tolist()]})
        df_30 = pd.DataFrame({"embds": [j for i in target_hl_30_embds for j in i.tolist()], "labels": [j for i in target_labels for j in i.tolist()]})
        
        b_id = str(uuid4())
        df_24.to_json("/Data2/Arun-UAV/NLP/vision_halu/haloc/embeddings/caption/llava_24/batch_" + str(b_id) + ".jsonl", lines=True, orient="records")
        df_30.to_json("/Data2/Arun-UAV/NLP/vision_halu/haloc/embeddings/caption/llava_30/batch_" + str(b_id) + ".jsonl", lines=True, orient="records")

        del input_ids, output_ids, attention_mask, image_tensor, ans_masks, expanded_input_ids, expanded_token_level_labels, expanded_ans_masks, target_hl_30_embds, target_hl_24_embds, target_labels, df_24, df_30
        torch.cuda.empty_cache()
        


def batch_eval_model(args):
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
    

    dataset_name="holoc_caption"
    collate_fn = collate_fn_builder(processor, None)
    dataloader = _initialize_dataloader(dataset_name=dataset_name, collate_fn=collate_fn, num_workers=32, batch_size=32)
    
    for batch in tqdm(dataloader, desc="storing embds"):
        generate_llava(batch, tokenizer, model, processor)

    # if your batch dict retains GPU tensors, also: del batch

    # 5) return unused cached blocks to the driver (optional but helpful)
    torch.cuda.empty_cache()


def eval_model(args):
    # Model
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

    test_data = "chair"
    
    if test_data =="chair":
    
        test_data = pd.read_csv("/Data2/Arun-UAV/NLP/vision_halu/benchmarks/chair_coco_500/chair_500_gemini_flash_2_5_des.csv")
        all_outputs = []
        for inx, row in tqdm(test_data.iterrows()):
            qs = "Please describe this image in detail."
            idx = row["image_id"]
            image_path = "/Data2/Arun-UAV/NLP/vision_halu/benchmarks/coco2024/val2014/COCO_val2014_" + str(idx).zfill(12) + ".jpg"
            descriptions = row["description"]
            
            
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_files = [image_path]
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            all_outputs.append(outputs)
        
        test_data["llava_1.5_all_img_token_add_des"] = all_outputs
        test_data.to_json("/Data2/Arun-UAV/NLP/vision_halu/testing_res/chair_llava_plus_des_embed_20_des.jsonl", lines=True, orient="records")

    elif test_data == "pope":
        pass


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

    batch_eval_model(args)
    # eval_model(args)
