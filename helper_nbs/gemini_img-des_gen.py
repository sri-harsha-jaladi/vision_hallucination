import os
import io
import cv2
import json
import pandas as pd
from tqdm import tqdm
from typing import List
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed

from google import genai
from google.oauth2.service_account import Credentials
from google.genai import types

# ================== CONFIG ==================
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
SERVICE_ACCOUNT_FILE = "/Data2/Arun-UAV/NLP/self-halu-detection/vertix_ai.json"
PROJECT_ID = "hazel-math-472314-h9"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

COCO_JSON = "/Data2/Arun-UAV/NLP/vision_halu/train_datasets/coco_sample_5000.json"
IMAGE_DIR = "/Data2/Arun-UAV/NLP/vision_halu/train_datasets/poc_5000_coco_images"
OUTPUT_JSON = "/Data2/Arun-UAV/NLP/vision_halu/train_datasets/coco_img_descriptions_parallel.json"

MAX_WORKERS = 4  # Adjust based on CPU count
# ============================================

# ---------- GEMINI CLIENT SETUP -------------
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)
# --------------------------------------------

# ---------- PROMPT & RESPONSE SCHEMA --------
class ImageDescription(BaseModel):
    image_description: str = Field(
        ..., description="Detailed description of the given image."
    )

PROMPT = """
You are a specialist in rich and precise scene understanding.
Given an input image, generate a comprehensive, contextually aware, and fluent description that captures all key visual elements, their relationships, emotions, and possible context or story.

Your description should go beyond short captions — it must resemble a paragraph of visual storytelling that includes:

Scene type: indoor/outdoor, environment, lighting, time of day
Objects and entities: names, counts, shapes, colors, materials
Actions and interactions: what the people or objects are doing
Spatial layout: foreground, background, relative positions
Emotions or atmosphere: tone, mood, aesthetics
Possible context: what might be happening or implied by the scene

Avoid generic or repetitive statements. Be vivid, factual, and coherent. Use natural language instead of bullet points.

Output JSON format:
{image_description: <full attached image description>}
"""
# --------------------------------------------


def get_image_description(img_path: str):
    """Worker function: generate detailed image description."""
    try:
        image = cv2.imread(img_path)
        if image is None:
            return {"file_name": os.path.basename(img_path), "error": "Image not found"}

        _, encoded_img = cv2.imencode(".jpg", image)
        img_bytes = io.BytesIO(encoded_img.tobytes()).getvalue()

        contents = [PROMPT, types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]

        structured_config = types.GenerateContentConfig(
            temperature=0.6,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=65535,
            response_schema=ImageDescription,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=structured_config,
        )

        parsed = response.parsed.model_dump()
        parsed["file_name"] = os.path.basename(img_path)
        return parsed

    except Exception as e:
        return {"file_name": os.path.basename(img_path), "error": str(e)}

# ---------- MAIN PARALLEL EXECUTION ----------
if __name__ == "__main__":
    coco_data = pd.read_json(COCO_JSON)
    img_files = coco_data["file_name"].tolist()[:200]
    img_list = os.listdir("/Data2/Arun-UAV/NLP/vision_halu/train_datasets/poc_5000_coco_images")

    print(f"🚀 Starting parallel processing for {len(img_files)} images...")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_image_description, os.path.join(IMAGE_DIR, f)): f
            for f in img_files if f in img_list
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            results.append(result)

    # Save all results to JSON
    with open(OUTPUT_JSON, "w") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"✅ Done! Saved {len(results)} image descriptions to {OUTPUT_JSON}")
