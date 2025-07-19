import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm

AUGMENTATIONS_PER_IMAGE = 10  # 1枚あたり生成する枚数

# ========== 拡張処理 ==========
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = random.uniform(5, 30)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, amount=0.01):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

def draw_random_lines(image, num_lines=3):
    img = image.copy()
    h, w = img.shape[:2]
    for _ in range(num_lines):
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        color = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
        thickness = random.randint(1, 2)
        cv2.line(img, pt1, pt2, color, thickness)
    return img

def apply_affine_distortion(image):
    rows, cols = image.shape[:2]
    pts1 = np.float32([[5, 5], [cols - 5, 5], [5, rows - 5]])
    shift = random.randint(-5, 5)
    pts2 = np.float32([[5 + shift, 5], [cols - 5 + shift, 5], [5, rows - 5 + shift]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (cols, rows))

def rotate_image(image, angle=None):
    if angle is None:
        angle = random.choice([-15, -10, 10, 15])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image):
    return cv2.flip(image, 1)

# ========== 全体水増し ==========
def generate_augmented_images(image):
    augmentations = []

    augmentations.append(rotate_image(image))
    augmentations.append(flip_image(image))
    augmentations.append(add_gaussian_noise(image))
    augmentations.append(add_salt_pepper_noise(image))
    augmentations.append(draw_random_lines(image))
    augmentations.append(apply_affine_distortion(image))
    augmentations.append(draw_random_lines(add_gaussian_noise(image)))
    augmentations.append(apply_affine_distortion(add_salt_pepper_noise(image)))
    augmentations.append(rotate_image(add_gaussian_noise(image)))
    augmentations.append(draw_random_lines(flip_image(image)))

    return augmentations[:AUGMENTATIONS_PER_IMAGE]


input_root = 'chars'
output_root = 'AugmentedChars'

for char_name in os.listdir(input_root):
    input_folder = os.path.join(input_root, char_name)
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_root, char_name)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob(os.path.join(input_folder, '*.png'))

    print(f"Processing {char_name}...")

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            continue

        # グレースケール → BGR に変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        augmented_images = generate_augmented_images(image)

        for idx, aug in enumerate(augmented_images):
            save_name = f"{base_name}_aug{idx}.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, aug)
