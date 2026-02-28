import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

def create_directories(subdirs):
    """Create all required directories if they do not exist."""
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)

def generate_healthy_base(size=(256, 256)):
    """
    Generate a base healthy tomato leaf image.
    Algorithm:
    1. Start with green background (RGB around [30, 120, 30])
    2. Add slight random green noise variation
    3. Add subtle darker vein-like lines using random thin lines
    4. Add small brightness variation per pixel
    5. Slight Gaussian blur to simulate camera softness
    """
    # 1. Start with green background
    base_color = np.array([30, 120, 30], dtype=np.uint8)
    img_array = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
    
    # 2. Add slight random green noise variation (-20 to 20)
    noise = np.random.randint(-20, 20, (size[1], size[0]), dtype=np.int16)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + noise, 0, 255).astype(np.uint8)
    
    # 4. Add small brightness variation per pixel
    brightness_noise = np.random.randint(-10, 10, (size[1], size[0], 3), dtype=np.int16)
    img_array = np.clip(img_array + brightness_noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, 'RGB')
    draw = ImageDraw.Draw(img)
    
    # 3. Add subtle darker vein-like lines
    num_veins = random.randint(10, 20)
    for _ in range(num_veins):
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        x2, y2 = x1 + random.randint(-50, 50), y1 + random.randint(-50, 50)
        draw.line([(x1, y1), (x2, y2)], fill=(20, 90, 20), width=random.randint(1, 3))
        
    # 5. Slight Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img

def add_lesions(img, size=(256, 256)):
    """
    Generate a diseased tomato leaf image.
    Algorithm:
    1. Start with same healthy base
    2. Add random brown/yellow circular lesions
    3. Add irregular edge variation
    4. Add random small speckle noise
    5. Slight blur
    """
    draw = ImageDraw.Draw(img)
    
    # 2. Add 3-10 lesions per image
    num_lesions = random.randint(3, 10)
    for _ in range(num_lesions):
        center_x = random.randint(20, size[0]-20)
        center_y = random.randint(20, size[1]-20)
        # Random radius 5-20 pixels
        radius = random.randint(5, 20)
        
        # Colors: brown [120,60,30] or yellow [180,180,40]
        if random.random() > 0.5:
            color = (120, 60, 30) # Brown
        else:
            color = (180, 180, 40) # Yellow
            
        # 3. Add irregular edge variation by drawing multiple overlapping slightly offset circles
        draw.ellipse([(center_x - radius, center_y - radius), 
                      (center_x + radius, center_y + radius)], fill=color)
        
        for _ in range(3):
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)
            sub_radius = radius - random.randint(1, 5)
            draw.ellipse([(center_x + offset_x - sub_radius, center_y + offset_y - sub_radius), 
                          (center_x + offset_x + sub_radius, center_y + offset_y + sub_radius)], fill=color)

    # 4. Add random small speckle noise
    img_array = np.array(img).astype(np.int16)
    speckle_noise = np.random.randint(-30, 30, (size[1], size[0], 3), dtype=np.int16)
    mask = np.random.random((size[1], size[0], 1)) > 0.95
    img_array = np.where(mask, img_array + speckle_noise, img_array)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, 'RGB')
    
    # 5. Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img

def generate_dataset():
    # Directories
    base_dir = "data/processed/triage"
    train_healthy_dir = os.path.join(base_dir, "train", "healthy")
    train_abnormal_dir = os.path.join(base_dir, "train", "abnormal")
    val_healthy_dir = os.path.join(base_dir, "val", "healthy")
    val_abnormal_dir = os.path.join(base_dir, "val", "abnormal")
    
    # Create all directories
    directories = [
        train_healthy_dir, train_abnormal_dir, 
        val_healthy_dir, val_abnormal_dir,
        "models", "runs"
    ]
    create_directories(directories)
    
    # Generate 20 healthy images
    for i in range(1, 21):
        img = generate_healthy_base()
        img.save(os.path.join(train_healthy_dir, f"healthy_{i:02d}.jpg"))
    print("Generated 20 healthy images")
    
    # Generate 20 abnormal images
    for i in range(1, 21):
        base_img = generate_healthy_base()
        abnormal_img = add_lesions(base_img)
        abnormal_img.save(os.path.join(train_abnormal_dir, f"abnormal_{i:02d}.jpg"))
    print("Generated 20 abnormal images")
    
    # Output final dataset path
    print(f"Final dataset path: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    generate_dataset()
