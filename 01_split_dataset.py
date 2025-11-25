import os
import random
import shutil
from pathlib import Path

random.seed(42) 
compressed_dataset = Path("compressed_dataset")
train_dir =  Path("train")
val_dir =  Path("val")

for country in compressed_dataset.iterdir():
    if not country.is_dir(): continue
    images = list(country.glob("*.jpg"))
    random.shuffle(images)
    split = int(0.8 * len(images))

    (train_dir / country.name).mkdir(exist_ok=True, parents=True)
    (val_dir / country.name).mkdir(exist_ok=True, parents=True)

    # use random to split the images randomly into train and val 
    # (80% for train, 20% for val)   
    for img in images[:split]:
        shutil.copy(img, train_dir / country.name / img.name)
    for img in images[split:]:
        shutil.copy(img, val_dir / country.name / img.name)