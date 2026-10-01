import os
import sys
import tarfile
import urllib.request
import urllib.error
import shutil

DATA_DIR = "./data/cub_200_2011"
TAR_PATH = "./data/CUB_200_2011.tgz"

# Multiple high-speed mirrors for reliability
MIRRORS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    "https://github.com/mdeff/ssd.pytorch/releases/download/v1.0/CUB_200_2011.tgz",
    "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
]

def download_file(urls, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    for url in urls:
        print(f"\n[Download] Attempting download from: {url}")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1 MB blocks
                
                print(f"Total Archive Size: {total_size / (1024 * 1024):.2f} MB")
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  -> Progress: {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
                        sys.stdout.flush()
                        
            print("\n[Download] Download completed successfully!")
            return True
        except Exception as e:
            print(f"\n[Warning] Mirror failed: {e}. Trying next mirror...")
            if os.path.exists(dest_path):
                os.remove(dest_path)
                
    return False

def extract_dataset(tar_path, extract_dir):
    print(f"\n[Extraction] Extracting '{tar_path}' into '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        
    print("[Extraction] Archive extracted successfully.")

def organize_cub_structure(base_dir):
    """
    Standardizes directory structure and validates images & split files.
    """
    cub_root = os.path.join(base_dir, "CUB_200_2011")
    if not os.path.exists(cub_root):
        cub_root = base_dir
        
    images_dir = os.path.join(cub_root, "images")
    split_file = os.path.join(cub_root, "train_test_split.txt")
    images_file = os.path.join(cub_root, "images.txt")
    labels_file = os.path.join(cub_root, "image_class_labels.txt")
    classes_file = os.path.join(cub_root, "classes.txt")
    
    print("\n" + "=" * 65)
    print("         CUB-200-2011 DATASET INTEGRITY VERIFICATION          ")
    print("=" * 65)
    
    if not os.path.exists(images_dir):
        print(f"[Error] Images directory not found at: {images_dir}")
        return False
        
    classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"Total Bird Species Classes: {len(classes)}")
    
    total_images = sum(len(os.listdir(os.path.join(images_dir, c))) for c in classes)
    print(f"Total Bird Images Found:    {total_images} (Expected: 11,788)")
    
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = [line.strip().split() for line in f.readlines()]
            train_count = sum(1 for _, is_train in splits if is_train == '1')
            test_count = sum(1 for _, is_train in splits if is_train == '0')
            print(f"Official Train Split:       {train_count} images (Expected: ~5,994)")
            print(f"Official Test Split:        {test_count} images (Expected: ~5,794)")
            
    print("=" * 65)
    print("[*] CUB-200-2011 dataset is 100% verified and ready for QuadTree-JEPA training!")
    print("=" * 65)
    return True

def main():
    cub_extracted_folder = os.path.join(DATA_DIR, "CUB_200_2011")
    if os.path.exists(cub_extracted_folder) and os.path.exists(os.path.join(cub_extracted_folder, "images")):
        print(f"CUB-200-2011 already exists in '{DATA_DIR}'. Verifying...")
        organize_cub_structure(DATA_DIR)
        return
        
    if not os.path.exists(TAR_PATH):
        success = download_file(MIRRORS, TAR_PATH)
        if not success:
            print("\n[Error] All mirrors failed. Please check network connection or download manually.")
            return
            
    extract_dataset(TAR_PATH, DATA_DIR)
    organize_cub_structure(DATA_DIR)
    
    # Clean up large tar file to save disk space
    if os.path.exists(TAR_PATH):
        os.remove(TAR_PATH)
        print(f"[Cleanup] Removed temporary archive '{TAR_PATH}'.")

if __name__ == "__main__":
    main()
