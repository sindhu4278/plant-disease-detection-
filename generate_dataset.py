import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Define output path
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Class names matching PlantVillage naming style
CLASSES = [
    "Apple___healthy",
    "Apple___Black_rot",
    "Corn___Common_rust",
    "Tomato___Bacterial_spot",
    "Tomato___healthy"
]

IMG_SIZE = 224
IMAGES_PER_CLASS = 220

def create_leaf_base(draw, shape_type, base_color):
    """Draws a basic leaf shape with veins on a PIL drawing context."""
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    
    if shape_type == "oval": # Apple leaf: simple oval/elliptical shape
        # Draw leaf body
        box = [center_x - 70, center_y - 85, center_x + 70, center_y + 85]
        draw.ellipse(box, fill=base_color, outline=(base_color[0]-30, base_color[1]-20, base_color[2]-30), width=2)
        
        # Stem
        draw.line([center_x, center_y + 85, center_x, center_y + 110], fill=(80, 50, 20), width=4)
        # Main vein
        draw.line([center_x, center_y - 85, center_x, center_y + 85], fill=(base_color[0]+30, base_color[1]+30, base_color[2]+10), width=3)
        # Side veins
        for i in range(-70, 70, 25):
            draw.line([center_x, center_y + i, center_x - 50, center_y + i - 20], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+10), width=1)
            draw.line([center_x, center_y + i, center_x + 50, center_y + i - 20], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+10), width=1)
            
    elif shape_type == "elongated": # Corn leaf: long and narrow
        # Draw corn leaf
        points = [
            (center_x, center_y - 100),   # Tip
            (center_x + 25, center_y - 30),
            (center_x + 35, center_y + 50),
            (center_x + 10, center_y + 100), # Base
            (center_x - 10, center_y + 100),
            (center_x - 35, center_y + 50),
            (center_x - 25, center_y - 30),
        ]
        draw.polygon(points, fill=base_color, outline=(base_color[0]-30, base_color[1]-20, base_color[2]-30), width=2)
        
        # Stem/Vein
        draw.line([center_x, center_y + 100, center_x, center_y + 115], fill=(100, 70, 30), width=5)
        # Long veins
        draw.line([center_x, center_y - 100, center_x, center_y + 100], fill=(base_color[0]+30, base_color[1]+30, base_color[2]+10), width=3)
        draw.line([center_x - 15, center_y - 50, center_x - 15, center_y + 100], fill=(base_color[0]+15, base_color[1]+15, base_color[2]+5), width=1)
        draw.line([center_x + 15, center_y - 50, center_x + 15, center_y + 100], fill=(base_color[0]+15, base_color[1]+15, base_color[2]+5), width=1)
        
    elif shape_type == "jagged": # Tomato leaf: compound, irregular/jagged border
        points = [
            (center_x, center_y - 95),   # Top lobe
            (center_x + 20, center_y - 70),
            (center_x + 10, center_y - 65),
            (center_x + 45, center_y - 40), # Middle right lobe
            (center_x + 25, center_y - 30),
            (center_x + 55, center_y + 10), # Bottom right lobe
            (center_x + 20, center_y + 20),
            (center_x + 15, center_y + 80), # Base
            (center_x - 15, center_y + 80),
            (center_x - 20, center_y + 20),
            (center_x - 55, center_y + 10), # Bottom left lobe
            (center_x - 25, center_y - 30),
            (center_x - 45, center_y - 40), # Middle left lobe
            (center_x - 10, center_y - 65),
            (center_x - 20, center_y - 70),
        ]
        draw.polygon(points, fill=base_color, outline=(base_color[0]-35, base_color[1]-20, base_color[2]-35), width=2)
        
        # Stem
        draw.line([center_x, center_y + 80, center_x, center_y + 105], fill=(70, 85, 30), width=4)
        # Veins
        draw.line([center_x, center_y - 95, center_x, center_y + 80], fill=(base_color[0]+25, base_color[1]+25, base_color[2]+5), width=2)
        draw.line([center_x, center_y - 40, center_x + 35, center_y - 45], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+5), width=1)
        draw.line([center_x, center_y - 40, center_x - 35, center_y - 45], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+5), width=1)
        draw.line([center_x, center_y + 10, center_x + 40, center_y + 5], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+5), width=1)
        draw.line([center_x, center_y + 10, center_x - 40, center_y + 5], fill=(base_color[0]+20, base_color[1]+20, base_color[2]+5), width=1)

def add_noise_and_background(img):
    """Applies random noise and background variations to make images look more realistic."""
    img_np = np.array(img).astype(np.float32)
    # Add random background color shift
    bg_r = random.randint(235, 250)
    bg_g = random.randint(230, 245)
    bg_b = random.randint(220, 235)
    
    # Mask where image is white (default background)
    mask = (img_np[:, :, 0] == 255) & (img_np[:, :, 1] == 255) & (img_np[:, :, 2] == 255)
    
    img_np[mask, 0] = bg_r
    img_np[mask, 1] = bg_g
    img_np[mask, 2] = bg_b
    
    # Add Gaussian/uniform noise
    noise = np.random.normal(0, 4, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

def generate_image(cls_name):
    # Create empty image (white background)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Base color variances
    r_val = random.randint(35, 75)
    g_val = random.randint(110, 160)
    b_val = random.randint(25, 55)
    base_color = (r_val, g_val, b_val)
    
    if "Apple" in cls_name:
        shape = "oval"
    elif "Corn" in cls_name:
        shape = "elongated"
        # Corn leaves are slightly lighter/yellow-green
        base_color = (r_val + 20, g_val - 10, b_val)
    else:
        shape = "jagged"
        
    create_leaf_base(draw, shape, base_color)
    
    # Add disease features
    if "Black_rot" in cls_name:
        # Apple Black rot: large brown spots with darker edges
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            sx = random.randint(85, 135)
            sy = random.randint(80, 140)
            radius = random.randint(6, 14)
            # Outer dark edge
            draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=(50, 30, 20))
            # Inner lighter brown
            inner_radius = int(radius * 0.7)
            draw.ellipse([sx - inner_radius, sy - inner_radius, sx + inner_radius, sy + inner_radius], fill=(120, 75, 45))
            
    elif "Common_rust" in cls_name:
        # Corn Common Rust: oblong orange-brown rust pustules
        num_spots = random.randint(12, 25)
        for _ in range(num_spots):
            sx = random.randint(100, 124)
            sy = random.randint(50, 170)
            length = random.randint(6, 12)
            width = random.randint(2, 4)
            # Draw elongated orange-brown spots
            draw.ellipse([sx - width, sy - length//2, sx + width, sy + length//2], fill=(185, 90, 25))
            
    elif "Bacterial_spot" in cls_name:
        # Tomato Bacterial spot: tiny black spots with yellow halos
        num_spots = random.randint(15, 30)
        for _ in range(num_spots):
            sx = random.randint(70, 150)
            sy = random.randint(60, 160)
            
            # Check if spot falls roughly inside the leaf (by reading the green channel intensity at the location)
            # Simple check to make spots appear mostly on leaf
            if abs(sx - IMG_SIZE//2) < 55 and abs(sy - IMG_SIZE//2) < 70:
                halo_radius = random.randint(4, 7)
                spot_radius = random.randint(1, 2)
                # Yellow halo
                draw.ellipse([sx - halo_radius, sy - halo_radius, sx + halo_radius, sy + halo_radius], fill=(210, 210, 60))
                # Dark spot center
                draw.ellipse([sx - spot_radius, sy - spot_radius, sx + spot_radius, sy + spot_radius], fill=(20, 20, 20))
                
    # Post-process with background texture, noise and slight blur for natural blending
    img = add_noise_and_background(img)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    
    return img

def main():
    print("Starting synthetic dataset generation...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    total_generated = 0
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"Generating {IMAGES_PER_CLASS} images for class '{cls}'...")
        
        for idx in range(IMAGES_PER_CLASS):
            img = generate_image(cls)
            img.save(os.path.join(cls_dir, f"leaf_{idx:03d}.jpg"), quality=95)
            total_generated += 1
            
    print(f"Dataset generation complete! Total images generated: {total_generated} in directory: {DATA_DIR}")

if __name__ == "__main__":
    main()
