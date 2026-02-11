# Create Multi-Label Dataset by Making Collage Images + labels.csv

import os
import random
import csv
from PIL import Image

BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-4_MultiLabel_Service"
IMG_DIR = os.path.join(BASE_DIR, "dataset", "images")

OUT_DIR = os.path.join(BASE_DIR, "dataset", "multilabel_images")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "labels.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# You already used: bike=0, bus=1, car=2 conceptually (same names)
CLASSES = ["bike", "bus", "car"]

def list_images_containing(keyword: str):
    files = []
    for f in os.listdir(IMG_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # use folder history is gone; so use random split: we will just pick from all images
            files.append(os.path.join(IMG_DIR, f))
    return files

all_imgs = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(all_imgs) < 30:
    raise RuntimeError("Not enough base images in dataset/images. Need at least ~30.")

def safe_open(path):
    img = Image.open(path).convert("RGB")
    return img

def make_collage(paths, out_path, size=(224,224)):
    # 2 or 3 images => grid collage
    W, H = size
    canvas = Image.new("RGB", (W, H))

    if len(paths) == 2:
        # left/right
        img1 = safe_open(paths[0]).resize((W//2, H))
        img2 = safe_open(paths[1]).resize((W//2, H))
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (W//2, 0))
    else:
        # 3 images: 2 on top, 1 bottom full
        img1 = safe_open(paths[0]).resize((W//2, H//2))
        img2 = safe_open(paths[1]).resize((W//2, H//2))
        img3 = safe_open(paths[2]).resize((W, H//2))
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (W//2, 0))
        canvas.paste(img3, (0, H//2))

    canvas.save(out_path, quality=90)

rows = []
random.seed(42)

# Create 30 multi-label (2 labels) + 10 multi-label (3 labels)
two_label_count = 30
three_label_count = 10

for i in range(two_label_count):
    # random 2 labels
    labels = random.sample(CLASSES, 2)
    img_paths = random.sample(all_imgs, 2)

    out_name = f"ml_2_{i:03d}.jpg"
    out_path = os.path.join(OUT_DIR, out_name)
    make_collage(img_paths, out_path)

    row = {
        "filename": out_name,
        "bike": 1 if "bike" in labels else 0,
        "bus":  1 if "bus" in labels else 0,
        "car":  1 if "car" in labels else 0,
    }
    rows.append(row)

for i in range(three_label_count):
    labels = ["bike", "bus", "car"]
    img_paths = random.sample(all_imgs, 3)

    out_name = f"ml_3_{i:03d}.jpg"
    out_path = os.path.join(OUT_DIR, out_name)
    make_collage(img_paths, out_path)

    row = {"filename": out_name, "bike": 1, "bus": 1, "car": 1}
    rows.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename","bike","bus","car"])
    writer.writeheader()
    writer.writerows(rows)

print("Created images:", len(rows))
print("Saved images to:", OUT_DIR)
print("Saved labels.csv to:", CSV_PATH)