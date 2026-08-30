import os
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CLASSES = [
    "Tomato___healthy",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Early_blight"
]

BASE_API_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/"
BASE_OUTPUT_DIR = "./data/plant_dataset"
NUM_TRAIN = 80
NUM_TEST = 20
TOTAL_PER_CLASS = NUM_TRAIN + NUM_TEST

def fetch_file_list(class_name):
    url = BASE_API_URL + class_name
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    # Filter for image files
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    image_files = [f for f in data if f['type'] == 'file' and any(f['name'].endswith(ext) for ext in valid_exts)]
    return image_files

def download_single_image(download_url, save_path):
    if os.path.exists(save_path):
        return
    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as resp, open(save_path, 'wb') as f:
        f.write(resp.read())

def prepare_dataset():
    random.seed(42)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    tasks = []
    
    for cls in CLASSES:
        print(f"\n[1/2] Fetching file catalog for: {cls}...")
        image_entries = fetch_file_list(cls)
        
        if len(image_entries) < TOTAL_PER_CLASS:
            print(f"Warning: Only {len(image_entries)} found for {cls}")
            selected = image_entries
        else:
            # Shuffle deterministically and pick 100 images
            random.shuffle(image_entries)
            selected = image_entries[:TOTAL_PER_CLASS]
            
        train_entries = selected[:NUM_TRAIN]
        test_entries = selected[NUM_TRAIN:TOTAL_PER_CLASS]
        
        train_dir = os.path.join(BASE_OUTPUT_DIR, "train", cls)
        test_dir = os.path.join(BASE_OUTPUT_DIR, "test", cls)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for item in train_entries:
            dest = os.path.join(train_dir, item['name'])
            tasks.append((item['download_url'], dest))
            
        for item in test_entries:
            dest = os.path.join(test_dir, item['name'])
            tasks.append((item['download_url'], dest))
            
    print(f"\n[2/2] Downloading {len(tasks)} images (Multi-threaded)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_single_image, url, path) for url, path in tasks]
        for i, future in enumerate(futures, 1):
            future.result()
            if i % 30 == 0 or i == len(tasks):
                print(f"  -> Downloaded {i}/{len(tasks)} images...")
                
    print("\n Download and 80:20 dataset split successfully created!")
    print(f"Location: {os.path.abspath(BASE_OUTPUT_DIR)}")
    print("Structure:")
    print(f"  ./data/plant_dataset/train/ (80 images x 3 classes = 240 images)")
    print(f"  ./data/plant_dataset/test/  (20 images x 3 classes = 60 images)")

if __name__ == "__main__":
    prepare_dataset()
