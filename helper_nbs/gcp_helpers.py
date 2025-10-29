import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import storage
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
SERVICE_ACCOUNT_FILE = "/Data2/Arun-UAV/NLP/new_cloud_coount.json"
BUCKET_NAME = "train_data_vision_1"
FOLDER_PATH = "/Data2/Arun-UAV/NLP/vision_halu/train_datasets/poc_5000_coco_images"
PREFIX = "coco_batch_1_15000/"   # Optional subfolder in bucket
MAX_WORKERS = 64                    # Tune to number of CPU cores
# ============================================

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# ---------- Helper Function ----------
def upload_single_image(file_path: str) -> tuple:
    """Worker function to upload a single file to GCS."""
    try:
        # Create a new GCS client inside each process
        credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(BUCKET_NAME)

        blob_name = os.path.join(PREFIX, os.path.basename(file_path))
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        return (file_path, gcs_uri, None)
    except Exception as e:
        return (file_path, None, str(e))

# ---------- Parallel Upload ----------
def upload_images_to_gcs_parallel(folder_path: str) -> pd.DataFrame:
    """Uploads all images in folder_path to GCS using multiple processes."""
    # Collect all image files
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_single_image, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading to GCS"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=["local_path", "gcs_uri", "error"])
    return df

# ---------- MAIN ----------
if __name__ == "__main__":
    df = upload_images_to_gcs_parallel(FOLDER_PATH)
    output_csv = "/Data2/Arun-UAV/NLP/vision_halu/train_datasets/coco_batch_1_15000_gcp_upload_urs.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Upload complete — results saved to {output_csv}")
    
    
