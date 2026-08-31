import os
import time
import json
import random
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# DATASET CONFIGURATION: TOMATO, APPLE, CORN (25,183 IMAGES)
# ==========================================
CLASSES = [
    # Apple (4 classes - 3,171 images)
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    
    # Corn (4 classes - 3,852 images)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    
    # Tomato (10 classes - 18,160 images)
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

BASE_OUTPUT_DIR = "./data/plant_dataset"
TRAIN_RATIO = 0.80  # 80% train, 20% test
MAX_WORKERS = 20
MAX_RETRIES = 3

RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"

def fetch_class_tree_shas():
    """
    Fetches git tree SHAs for all category folders in raw/color
    to retrieve all files without being capped at 1,000 files by REST contents API.
    """
    url = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        entries = json.loads(resp.read().decode())
    return {e['name']: e['sha'] for e in entries if e['type'] == 'dir'}

def fetch_all_files_for_class(cls_name, tree_sha):
    """
    Fetches the full, un-truncated list of image filenames for a specific class using its Git tree SHA.
    """
    url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/git/trees/{tree_sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        tree_data = json.loads(resp.read().decode())
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    filenames = [
        item['path'] for item in tree_data.get('tree', []) 
        if item['type'] == 'blob' and any(item['path'].endswith(ext) for ext in valid_exts)
    ]
    return cls_name, filenames

def download_single_image(url, save_path, retries=MAX_RETRIES):
    """
    Downloads an image file with automatic retry logic and skip if already downloaded.
    """
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return True
        
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(save_path, 'wb') as f:
                f.write(resp.read())
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(0.5 * (attempt + 1))
    return False

def prepare_dataset():
    random.seed(42)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("   PLANTVILLAGE DATASET DOWNLOADER: TOMATO, APPLE, CORN (25,183 IMAGES)  ")
    print("=" * 70)
    print(f"Target Directory: {os.path.abspath(BASE_OUTPUT_DIR)}")
    print(f"Train/Test Split: {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%\n")
    
    # 1. Fetch directory metadata
    print("[Step 1/3] Fetching class directory metadata from GitHub...")
    tree_shas = fetch_class_tree_shas()
    
    # 2. Build full file lists for all 18 classes
    print("[Step 2/3] Fetching complete catalog for all 18 categories...")
    catalog = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_all_files_for_class, cls, tree_shas[cls]): cls for cls in CLASSES if cls in tree_shas}
        for future in as_completed(futures):
            cls, filenames = future.result()
            catalog[cls] = filenames
            print(f"  • {cls:<48} : {len(filenames):>5,} images")
            
    total_images = sum(len(files) for files in catalog.values())
    print(f"\n Total Cataloged: {total_images:,} images across {len(catalog)} classes.\n")
    
    # 3. Create download tasks with train/test stratified split
    download_tasks = []
    for cls, filenames in catalog.items():
        shuffled = filenames.copy()
        random.shuffle(shuffled)
        
        n_train = int(len(shuffled) * TRAIN_RATIO)
        train_files = shuffled[:n_train]
        test_files = shuffled[n_train:]
        
        train_dir = os.path.join(BASE_OUTPUT_DIR, "train", cls)
        test_dir = os.path.join(BASE_OUTPUT_DIR, "test", cls)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for fname in train_files:
            enc_cls = urllib.parse.quote(cls)
            enc_file = urllib.parse.quote(fname)
            raw_url = f"{RAW_BASE_URL}/{enc_cls}/{enc_file}"
            save_path = os.path.join(train_dir, fname)
            download_tasks.append((raw_url, save_path))
            
        for fname in test_files:
            enc_cls = urllib.parse.quote(cls)
            enc_file = urllib.parse.quote(fname)
            raw_url = f"{RAW_BASE_URL}/{enc_cls}/{enc_file}"
            save_path = os.path.join(test_dir, fname)
            download_tasks.append((raw_url, save_path))
            
    print(f"[Step 3/3] Downloading {len(download_tasks):,} images (Multi-threaded with {MAX_WORKERS} workers)...")
    start_time = time.time()
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_image, url, path): (url, path) for url, path in download_tasks}
        for future in as_completed(futures):
            success = future.result()
            completed += 1
            if not success:
                failed += 1
                
            if completed % 500 == 0 or completed == len(download_tasks):
                elapsed = time.time() - start_time
                img_per_sec = completed / max(elapsed, 0.001)
                remaining = (len(download_tasks) - completed) / max(img_per_sec, 0.001)
                pct = (completed / len(download_tasks)) * 100.0
                print(f"  -> Progress: {completed:>6,}/{len(download_tasks):,} ({pct:5.1f}%) | Speed: {img_per_sec:5.1f} img/s | ETA: {remaining/60:4.1f} min | Failed: {failed}")
                
    total_time = (time.time() - start_time) / 60
    print(f"\n Download complete in {total_time:.2f} minutes!")
    print(f"Successfully downloaded: {completed - failed:,} images (Failed: {failed})")
    print(f"Dataset location: {os.path.abspath(BASE_OUTPUT_DIR)}")

if __name__ == "__main__":
    prepare_dataset()
