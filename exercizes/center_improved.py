import cv2

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "DATA/elevator.jpg"
RESIZE_WIDTH = 1024
RESIZE_HEIGHT = 800

CENTER_MARGIN_RATIO = 0.10
CENTER_PADDING = 0.25

BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 3

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image could not be loaded. Check the file path.")
    exit()

# -----------------------------
# Resize image
# -----------------------------
img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

height, width, channels = img.shape

# -----------------------------
# Calculate center region
# -----------------------------
center_x = width // 2
center_y = height // 2

center_margin = int(width * CENTER_MARGIN_RATIO)

CENTER_PADDING = max(0.0, min(CENTER_PADDING, 0.49))

left_x = center_x - center_margin
right_x = center_x + center_margin

top_y = int(height * CENTER_PADDING)
bottom_y = int(height * (1 - CENTER_PADDING))

# -----------------------------
# Draw center box
# -----------------------------
cv2.rectangle(img, (left_x, top_y), (right_x, bottom_y), BOX_COLOR, BOX_THICKNESS)

# Optional center line
cv2.line(img, (center_x, 0), (center_x, height), (255, 0, 0), 2)

# -----------------------------
# Print information
# -----------------------------
print("Image width:", width)
print("Image height:", height)
print("Center x:", center_x)
print("Center y:", center_y)
print("Left x:", left_x)
print("Right x:", right_x)
print("Top y:", top_y)
print("Bottom y:", bottom_y)

# -----------------------------
# Display result
# -----------------------------
cv2.imshow("Centered Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()