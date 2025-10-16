import os
import json
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


# -------- CONFIG --------
OUTPUT_DIR = '/Data2/Arun-UAV/NLP/vision_halu/train_datasets/poc_5000_coco_images'
SAMPLES_JSON = '/Data2/Arun-UAV/NLP/vision_halu/train_datasets/coco_batch_1_10000.json'
MAX_WORKERS = 64      # Number of parallel processes (tune to CPU cores)
TIMEOUT = 10           # seconds
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load samples
with open(SAMPLES_JSON, 'r') as f:
    samples = json.load(f)

df = pd.read_json(SAMPLES_JSON)
samples = df.to_dict("records")

def download_image(sample):
    """Download a single image with error handling."""
    img_url = sample.get('coco_url')
    if not img_url:
        return f"⚠️ No URL for {sample.get('file_name', 'unknown')}"

    img_path = os.path.join(OUTPUT_DIR, sample['file_name'])
    if os.path.exists(img_path):
        return f"✅ Already exists: {sample['file_name']}"

    try:
        response = requests.get(img_url, stream=True, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(img_path, 'wb') as f_out:
                for chunk in response.iter_content(1024):
                    f_out.write(chunk)
            return f"✅ Downloaded: {sample['file_name']}"
        else:
            return f"❌ HTTP {response.status_code} for {img_url}"
    except Exception as e:
        return f"⚠️ Error {e} for {img_url}"

print(samples[0])
# -------- PARALLEL EXECUTION --------
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_image, s) for s in samples]

    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        result = f.result()
        if result.startswith("❌") or result.startswith("⚠️"):
            print(result)

print("✅ Done! All sampled images downloaded.")


# import requests
# import os

# def download_image(url: str, file_path: str = None):
#     """
#     Download an image from a given URL and save it locally.
    
#     Args:
#         url (str): The image URL.
#         save_dir (str): Directory to save the image (default current folder).
#         filename (str): Optional filename. If None, inferred from the URL.
    
#     Returns:
#         str: The full path to the saved image.
#     """
#     try:
#         # Get image content
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
    
#         with open(file_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
        
#         print(f"✅ Image saved to: {file_path}")
#         return file_path
    
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Failed to download image: {e}")
#         return None