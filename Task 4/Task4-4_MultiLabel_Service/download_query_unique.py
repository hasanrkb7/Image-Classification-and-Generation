# Download One Query With Unique Filenames (Safe)

import sys
import time
import os
import uuid
from bing_image_downloader import downloader

BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-4_MultiLabel_Service"
ROOT_DIR = os.path.join(BASE_DIR, "dataset", "images_downloads")
FINAL_DIR = os.path.join(BASE_DIR, "dataset", "images")

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print('Usage: python download_query_unique.py "your search query"')
    raise SystemExit

query = sys.argv[1]

LIMIT = 15
TIMEOUT = 60
RETRIES = 3

# Download into a unique subfolder
unique_folder = f"{query}__{uuid.uuid4().hex[:8]}"
out_dir = os.path.join(ROOT_DIR, unique_folder)

for attempt in range(1, RETRIES + 1):
    try:
        print(f"\nAttempt {attempt}/{RETRIES} -> {query}")
        downloader.download(
            query,
            limit=LIMIT,
            output_dir=out_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=TIMEOUT,
            verbose=True
        )
        print("Downloaded into:", out_dir)
        break
    except Exception as e:
        print("Error:", e)
        if attempt < RETRIES:
            time.sleep(5)
        else:
            raise

# Move all images to FINAL_DIR with unique names
for root, _, files in os.walk(out_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            new_name = f"{uuid.uuid4().hex}{ext}"
            dst = os.path.join(FINAL_DIR, new_name)
            os.replace(src, dst)

print("Done:", query)
print("Total images now:", len([x for x in os.listdir(FINAL_DIR) if x.lower().endswith(('.jpg','.jpeg','.png'))]))
