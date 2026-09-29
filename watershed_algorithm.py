import cv2
import numpy as np
from matplotlib import cm

# --------------------------------------------------
# Load image
# --------------------------------------------------
image_path = "/home/huseyin/opencv_env/DATA/road_image.jpg"
img_road = cv2.imread(image_path)

if img_road is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

road_copy = img_road.copy()

# Marker image for watershed
marker_image = np.zeros(img_road.shape[:2], dtype=np.int32)

# Segmented output
segments = np.zeros(img_road.shape, dtype=np.uint8)

# --------------------------------------------------
# Colors
# --------------------------------------------------
def create_color(i):
    rgb = (np.array(cm.tab10(i)[:3]) * 255).astype(int)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

colors = [create_color(i) for i in range(10)]

# --------------------------------------------------
# Global variables
# --------------------------------------------------
n_marker = 10
current_marker = 1
marks_updated = False

# --------------------------------------------------
# Mouse callback
# --------------------------------------------------
def mousecallback(event, x, y, flags, param):
    global marks_updated, marker_image, road_copy, current_marker

    if event == cv2.EVENT_LBUTTONDOWN:
        # draw marker label into watershed marker image
        cv2.circle(marker_image, (x, y), 10, current_marker, -1)

        # draw visible color on displayed image
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)

        marks_updated = True

# --------------------------------------------------
# Window
# --------------------------------------------------
cv2.namedWindow("Road Image")
cv2.setMouseCallback("Road Image", mousecallback)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    cv2.imshow("Road Image", road_copy)
    cv2.imshow("Watershed Segments", segments)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:   # ESC
        break

    elif k == ord('c'):
        road_copy = img_road.copy()
        marker_image = np.zeros(img_road.shape[:2], dtype=np.int32)
        segments = np.zeros(img_road.shape, dtype=np.uint8)
        marks_updated = False
        print("Cleared")

    elif ord('1') <= k <= ord('9'):
        current_marker = int(chr(k))
        print(f"Current marker: {current_marker}")

    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(img_road, marker_image_copy)

        segments = np.zeros(img_road.shape, dtype=np.uint8)

        for color_ind in range(1, n_marker):
            segments[marker_image_copy == color_ind] = colors[color_ind]

        # watershed boundaries
        segments[marker_image_copy == -1] = (0, 0, 255)

        marks_updated = False

cv2.destroyAllWindows()