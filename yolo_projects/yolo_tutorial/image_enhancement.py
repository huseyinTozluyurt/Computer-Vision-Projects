import cv2
import numpy as np
import os


def resize_image(img, target_width=None, target_height=None, scale=None):
    """
    Resize image by target size or scale.
    """
    h, w = img.shape[:2]

    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if target_width is not None and target_height is not None:
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    return img


def denoise_image(img, strength=8):
    """
    Remove color noise while preserving edges.
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21
    )


def adjust_brightness_contrast(img, brightness=10, contrast=1.15):
    """
    contrast > 1 increases contrast.
    brightness > 0 increases brightness.
    """
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def gamma_correction(img, gamma=1.05):
    """
    gamma < 1 brightens shadows.
    gamma > 1 darkens image.
    """
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Local contrast enhancement using LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))

    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def enhance_saturation(img, saturation_scale=1.10):
    """
    Slightly improve color saturation.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    h, s, v = cv2.split(hsv)

    s = np.clip(s * saturation_scale, 0, 255)

    hsv_enhanced = cv2.merge((h, s, v)).astype(np.uint8)

    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


def unsharp_mask(img, amount=1.4, blur_strength=2.0):
    """
    Natural-looking sharpening.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), blur_strength)

    sharpened = cv2.addWeighted(
        img,
        amount,
        blurred,
        1 - amount,
        0
    )

    return sharpened


def detail_enhancement(img, sigma_s=10, sigma_r=0.15):
    """
    OpenCV detail enhancement filter.
    Can make image look more stylized.
    """
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)


def edge_preserving_smooth(img, sigma_s=30, sigma_r=0.2):
    """
    Smooths noise while keeping edges.
    Useful before sharpening.
    """
    return cv2.edgePreservingFilter(
        img,
        flags=1,
        sigma_s=sigma_s,
        sigma_r=sigma_r
    )


def enhance_image_full_pipeline(
    input_path,
    output_path,
    target_width=None,
    target_height=None,
    scale=None,
    use_resize=True,
    use_denoise=True,
    use_edge_preserving=True,
    use_brightness_contrast=True,
    use_gamma=True,
    use_clahe=True,
    use_saturation=True,
    use_detail_enhancement=False,
    use_sharpen=True
):
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    original = img.copy()

    # 1. Resize / upscale
    if use_resize:
        img = resize_image(
            img,
            target_width=target_width,
            target_height=target_height,
            scale=scale
        )

    # 2. Denoise
    if use_denoise:
        img = denoise_image(img, strength=8)

    # 3. Edge-preserving smoothing
    if use_edge_preserving:
        img = edge_preserving_smooth(img, sigma_s=25, sigma_r=0.15)

    # 4. Brightness and contrast
    if use_brightness_contrast:
        img = adjust_brightness_contrast(
            img,
            brightness=8,
            contrast=1.12
        )

    # 5. Gamma correction
    if use_gamma:
        img = gamma_correction(img, gamma=1.03)

    # 6. CLAHE local contrast
    if use_clahe:
        img = apply_clahe(
            img,
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

    # 7. Saturation improvement
    if use_saturation:
        img = enhance_saturation(
            img,
            saturation_scale=1.08
        )

    # 8. Optional detail enhancement
    if use_detail_enhancement:
        img = detail_enhancement(
            img,
            sigma_s=10,
            sigma_r=0.12
        )

    # 9. Final sharpening
    if use_sharpen:
        img = unsharp_mask(
            img,
            amount=1.35,
            blur_strength=1.5
        )

    cv2.imwrite(output_path, img)

    print(f"Saved enhanced image to: {output_path}")

    return original, img


if __name__ == "__main__":
    input_image = "cat6.jpeg"
    output_image = "enhanced_output.jpeg"

    enhance_image_full_pipeline(
        input_path=input_image,
        output_path=output_image,

        # Option 1: upscale by scale
        scale=2.0,

        # Option 2: use exact size instead
        # target_width=1920,
        # target_height=1080,

        use_resize=True,
        use_denoise=True,
        use_edge_preserving=True,
        use_brightness_contrast=True,
        use_gamma=True,
        use_clahe=True,
        use_saturation=True,
        use_detail_enhancement=False,
        use_sharpen=True
    )