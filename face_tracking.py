import cv2
import numpy as np


def main():
    # Load Haar cascade
    cascade_path = "DATA/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load cascade file from: {cascade_path}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame from webcam.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(face_rects) == 0:
        print("Error: No face detected in the initial frame.")
        cap.release()
        return

    # Choose the largest detected face
    face_rects = sorted(face_rects, key=lambda r: r[2] * r[3], reverse=True)
    face_x, face_y, w, h = face_rects[0]
    track_window = (face_x, face_y, w, h)

    # Extract ROI for histogram
    roi = frame[face_y:face_y + h, face_x:face_x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask low-light / low-saturation pixels
    mask = cv2.inRange(hsv_roi, np.array((0, 30, 32)), np.array((180, 255, 255)))

    # Histogram on Hue channel
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Termination criteria
    term_crit = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        1
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame could not be read.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Back projection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply CamShift instead of MeanShift
        ret_camshift, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw rotated tracking box
        pts = cv2.boxPoints(ret_camshift)
        pts = np.int32(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        cv2.imshow("Face Tracking", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()