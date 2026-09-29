import cv2
import numpy as np


def compute_hsv_range_from_reference(ref_bgr, low_percentile=5, high_percentile=95,
                                     h_margin=10, s_margin=40, v_margin=40):
    """
    Estimate an HSV color range from the reference image of the coat/jacket.

    Assumption:
    - The reference image mostly contains the coat color.
    - Background inside the reference image is limited.

    Returns:
        lower_hsv, upper_hsv
    """
    hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)

    # Flatten channels
    h = hsv[:, :, 0].reshape(-1)
    s = hsv[:, :, 1].reshape(-1)
    v = hsv[:, :, 2].reshape(-1)

    # Ignore very dark / unsaturated pixels a bit to reduce background effect
    valid = (s > 30) & (v > 30)
    if np.sum(valid) < 100:
        valid = np.ones_like(h, dtype=bool)

    h_valid = h[valid]
    s_valid = s[valid]
    v_valid = v[valid]

    h_low = int(np.percentile(h_valid, low_percentile)) - h_margin
    h_high = int(np.percentile(h_valid, high_percentile)) + h_margin

    s_low = int(np.percentile(s_valid, low_percentile)) - s_margin
    s_high = int(np.percentile(s_valid, high_percentile)) + s_margin

    v_low = int(np.percentile(v_valid, low_percentile)) - v_margin
    v_high = int(np.percentile(v_valid, high_percentile)) + v_margin

    lower = np.array([
        max(0, h_low),
        max(0, s_low),
        max(0, v_low)
    ], dtype=np.uint8)

    upper = np.array([
        min(179, h_high),
        min(255, s_high),
        min(255, v_high)
    ], dtype=np.uint8)

    return lower, upper


def build_mask(frame_bgr, lower_hsv, upper_hsv):
    """
    Build a binary mask for the coat color and refine it with morphology.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Morphological operations
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Fill small holes inside detected cloak region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Slightly enlarge the region
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def main():
    # Change this path to your coat/jacket image
    cloak_image_path = "ceket.jpeg"

    ref_img = cv2.imread(cloak_image_path)
    if ref_img is None:
        print(f"Error: Could not read reference image: {cloak_image_path}")
        return

    lower_hsv, upper_hsv = compute_hsv_range_from_reference(ref_img)

    print("Estimated HSV range from reference image:")
    print("Lower HSV:", lower_hsv)
    print("Upper HSV:", upper_hsv)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    prev_frame = cv2.flip(prev_frame, 1)

    print("\nControls:")
    print("  q : quit")
    print("  p : toggle previous-frame cloak mode")
    print("  b : capture static background frame")
    print("  s : show current mask")
    print()

    use_previous_frame = True
    show_mask = False
    static_background = prev_frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Detect the coat/cloak region
        mask = build_mask(frame, lower_hsv, upper_hsv)

        # Optional extra smoothing
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Convert to boolean mask
        mask_binary = mask > 127
        mask_3ch = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)

        # Choose the source of background pixels
        if use_previous_frame:
            background_source = prev_frame
            mode_text = "Mode: previous frame"
        else:
            background_source = static_background
            mode_text = "Mode: static background"

        output = frame.copy()
        output[mask_3ch] = background_source[mask_3ch]

        # Put info text
        cv2.putText(output, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, "q: quit | p: prev/static | b: capture bg | s: mask",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Invisibility Cloak with Morphology", output)

        if show_mask:
            cv2.imshow("Mask", mask)
        else:
            cv2.destroyWindow("Mask")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            use_previous_frame = not use_previous_frame
        elif key == ord('b'):
            static_background = frame.copy()
            print("Static background captured.")
        elif key == ord('s'):
            show_mask = not show_mask

        # Update previous frame at the end
        prev_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()