from typing import List, Dict, Union
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from llava.mm_utils import (tokenizer_image_token)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates


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

    def format_text(self, text: str):
        if self.model_config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

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

    def get_processed_tokens_batch(self, batch_text: List[str], images: Union[List[str], List[Image.Image]]):
        prompt = [self.format_text(text) for text in batch_text]
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

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids
    
    
def collate_fn_builder(processor=None, tokenizer=None):
    def collate_fn(batch):
        bkeys = ["question", "answer", "question_id", "image", "image_id", "image_path", "data_type"]
        processed_batch = {bkey: [example[bkey] for example in batch] for bkey in bkeys if bkey in batch[0]}

        


        batch_images, batch_text = processor.get_processed_tokens_batch(
            [example["question"] for example in batch],
            [example["image_path"] for example in batch],
        )

        processed_batch["image_tensors"] = batch_images
        processed_batch["input_ids"] = batch_text
            
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
            "data_type": content.get("data_type", "no_datatype")
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
    