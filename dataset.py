import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class QuadtreeImageDataset(Dataset):
    """
    Production-ready dataset pipeline for Quadtree-JEPA.
    Loads real images from a local directory, scales them to 504x504,
    and applies standard vision model channel normalization.
    """
    def __init__(self, image_dir, target_size=504):
        self.image_dir = image_dir
        self.target_size = target_size
        
        # Get list of all valid image files in the directory
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in self.valid_extensions
        ]
        
        if len(self.image_paths) == 0:
            raise RuntimeWarning(f"No valid images found in directory: {image_dir}")

        # Strict ImageNet normalization parameters for stable latent representation learning
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return self.image_paths.length if hasattr(self.image_paths, 'length') else len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Open image and force RGB conversion to eliminate alpha channel or grayscale mismatches
            with Image.open(img_path) as raw_img:
                rgb_img = raw_img.convert('RGB')
                tensor_img = self.transform(rgb_img)
            return tensor_img
        except Exception as e:
            print(f"Warning: Failed to load image {img_path} due to error: {e}. Returning fallback zero tensor.")
            return torch.zeros(3, self.target_size, self.target_size)