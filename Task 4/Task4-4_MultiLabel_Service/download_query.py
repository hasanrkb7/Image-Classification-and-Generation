# Download One Query (Retry + Longer Timeout)

import sys
import time
from bing_image_downloader import downloader
import os

BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-4_MultiLabel_Service"
IMG_DIR = os.path.join(BASE_DIR, "dataset", "images")
os.makedirs(IMG_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print('Usage: python download_query.py "your search query"')
    raise SystemExit

query = sys.argv[1]

LIMIT = 15
TIMEOUT = 60
RETRIES = 3

for attempt in range(1, RETRIES + 1):
    try:
        print(f"\nAttempt {attempt}/{RETRIES} -> {query}")
        downloader.download(
            query,
            limit=LIMIT,
            output_dir=IMG_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=TIMEOUT,
            verbose=True
        )
        print("Done:", query)
        break
    except Exception as e:
        print("Error:", e)
        if attempt < RETRIES:
            time.sleep(5)
        else:
            print("Failed after retries.")
            raise
