import cv2
import numpy as np


def white_balance_gray_world(img):
    result = img.astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_low_light_noisy_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError("Image could not be loaded.")

    # 1. Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=12,
        hColor=12,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. White balance
    balanced = white_balance_gray_world(denoised)

    # 3. CLAHE in LAB color space
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Slight saturation improvement
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    s = np.clip(s * 1.08, 0, 255)
    v = np.clip(v * 1.03, 0, 255)

    hsv = cv2.merge((h, s, v)).astype(np.uint8)
    color_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 5. Unsharp masking
    blurred = cv2.GaussianBlur(color_enhanced, (0, 0), 1.5)

    sharpened = cv2.addWeighted(
        color_enhanced,
        1.35,
        blurred,
        -0.35,
        0
    )

    # 6. Save output
    cv2.imwrite(output_path, sharpened)


enhance_low_light_noisy_image(
    "cat6.jpeg",
    "cat6_enhanced.jpg"
)