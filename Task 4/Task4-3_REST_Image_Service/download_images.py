from bing_image_downloader import downloader
import os

BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-3_REST_Image_Service"
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# keyword -> target folder name
classes = {
    "car vehicle photo": "car",
    "motorcycle bike vehicle photo": "bike",
    "bus vehicle photo": "bus"
}

LIMIT_PER_CLASS = 80   # you can change to 60 if download is slow

for query, folder in classes.items():
    print(f"\nDownloading: {query} -> {folder}")
    out_dir = os.path.join(RAW_DIR, folder)

    # bing-image-downloader creates its own folder using the query.
    downloader.download(
        query,
        limit=LIMIT_PER_CLASS,
        output_dir=out_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )

print("\nDone.")