import os
import cv2
import numpy as np
import random

DATASET_DIR = os.path.join("data", "dataset")

def is_reasonable_face(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return False
    mean = img.mean()
    return 30 < mean < 230 # Relaxed for deeper training

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

def apply_perspective(img: np.ndarray) -> np.ndarray:
    """Simulates a camera tilt or head tilt."""
    h, w = img.shape[:2]
    src = np.float32([[0,0], [w,0], [0,h], [w,h]])
    # Random jitter
    j = random.randint(3, 10)
    dst = np.float32([[j,j], [w-j,j], [j,h-j], [w-j,h-j]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w,h), borderMode=cv2.BORDER_REPLICATE)

def augment_image(path: str, student_id: str, base_index: str) -> None:
    img = cv2.imread(path) # Read in BGR for color augmentation
    if img is None: return
    h_orig, w_orig = img.shape[:2]
    if h_orig < 60: return 

    basename = f"user.{student_id}.{base_index}"
    
    # Ultra-Deep Target: ~300+ variants per image
    count = 0
    angles = [-25, -15, -5, 0, 5, 15, 25]
    brightness = [-50, -25, 0, 25, 50]
    contrasts = [0.7, 1.0, 1.3]
    
    for angle in angles:
        rotated = rotate_image(img, angle)
        # Randomly apply perspective tilt
        if random.random() > 0.6:
            rotated = apply_perspective(rotated)
            
        for beta in brightness:
            for alpha in contrasts:
                count += 1
                variant = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
                
                # Dynamic effects
                if count % 10 == 0:
                    variant = cv2.GaussianBlur(variant, (3, 3), 0)
                if count % 15 == 0:
                    # Noise injection
                    noise = np.random.randint(0, 50, (h_orig, w_orig, 3), dtype='uint8')
                    variant = cv2.add(variant, noise)

                out_path = os.path.join(DATASET_DIR, f"{basename}_v{count}.jpg")
                cv2.imwrite(out_path, variant)
                
                # Symmetry/Flips
                if count % 2 == 0:
                    flipped = cv2.flip(variant, 1)
                    cv2.imwrite(os.path.join(DATASET_DIR, f"{basename}_f{count}.jpg"), flipped)

def main(status_callback=None) -> None:
    if not os.path.isdir(DATASET_DIR): return

    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".jpg") and "_" not in f]
    if not files: return

    msg = f"Deep-Training Initialized. Expanding {len(files)} base images into Ultra-Robust dataset..."
    if status_callback: status_callback(msg)
    
    for i, filename in enumerate(files):
        parts = filename.split(".")
        if len(parts) < 3: continue
        student_id, idx = parts[1], parts[2]
        
        if status_callback:
            status_callback(f"Generating Ultra-Deep patterns for {student_id} ({i+1}/{len(files)})...")
            
        augment_image(os.path.join(DATASET_DIR, filename), student_id, idx)

    if status_callback: status_callback("Ultra-Deep Data Generation Complete.")

if __name__ == "__main__":
    main()
