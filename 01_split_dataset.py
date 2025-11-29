## PEPROCESSING + DATASET SPLIT

import os
import random
import shutil
from pathlib import Path
from PIL import Image
import kagglehub
from tqdm.auto import tqdm

random.seed(42)

source_root = Path(kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")) / "compressed_dataset"
train_dir = Path("train")
val_dir = Path("val")
test_dir = Path("test")

# clear folders if they already exist
for d in tqdm([train_dir, val_dir, test_dir], desc="Clearing directories..."):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

countries = {}
for country in tqdm(list(source_root.iterdir()), desc="Filtering countries..."):
    if not country.is_dir():
        continue
    images = list(country.glob("*.jpg")) + list(country.glob("*.jpeg")) + list(country.glob("*.png"))
    valid_images = []
    for img in images:
        try:
            with Image.open(img) as im:
                w, h = im.size
            if w == 1030 and h == 703: # skip images that have dimensions 1030x703 (minority)
                print(f"skipping image from {country.name}")
                continue
            valid_images.append(img)
        except:
            continue
    if len(valid_images) <= 1000: # skip countries with 1000 or fewer images
        continue
    countries[country.name] = valid_images

for country_name, images in tqdm(countries.items(), desc="Creating splits..."):
    random.shuffle(images)
    n = len(images)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    splits = [
        (train_dir, images[:n_train]),
        (val_dir, images[n_train:n_train + n_val]),
        (test_dir, images[n_train + n_val:n_train + n_val + n_test]),
    ]

    for base_dir, subset in splits:
        if not subset:
            continue
        country_dir = base_dir / country_name
        country_dir.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, country_dir / img.name)