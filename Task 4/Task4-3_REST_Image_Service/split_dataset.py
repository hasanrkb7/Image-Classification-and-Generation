import os
import shutil
import random

BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-3_REST_Image_Service"
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")

SPLIT_RATIO = 0.8

classes = ["car", "bike", "bus"]

for cls in classes:
    raw_path = os.path.join(RAW_DIR, cls)
    train_path = os.path.join(TRAIN_DIR, cls)
    val_path = os.path.join(VAL_DIR, cls)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    images = os.listdir(raw_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(raw_path, img), os.path.join(train_path, img))

    for img in val_images:
        shutil.copy(os.path.join(raw_path, img), os.path.join(val_path, img))

    print(f"{cls}: {len(train_images)} train, {len(val_images)} validation")

print("\nDataset split complete.")
