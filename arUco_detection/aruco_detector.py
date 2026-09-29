import cv2
import numpy as np


def get_aruco_dictionary():
    """
    Recommended dictionary for your indoor delivery robot.
    DICT_6X6_250 gives enough unique IDs for tables, rooms, waypoints,
    charging station, kitchen, etc.
    """
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


#def create_aruco_detector():
#    aruco_dict = get_aruco_dictionary()
#
#    parameters = cv2.aruco.DetectorParameters()
#
#    # Improves corner precision. Useful later for pose estimation.
#    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

#    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#    return detector


def create_aruco_detector():
    aruco_dict = get_aruco_dictionary()

    parameters = cv2.aruco.DetectorParameters()

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Helps with markers appearing at different scales/distances
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10

    # Allows smaller markers to be considered
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0

    # Helps reject bad candidates but keep valid distant markers
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.02

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    return detector




def draw_marker_info(frame, corners, ids):
    frame_height, frame_width = frame.shape[:2]
    image_center_x = frame_width // 2
    image_center_y = frame_height // 2

    # Draw image center line
    cv2.line(frame, (image_center_x, 0), (image_center_x, frame_height), (255, 255, 255), 1)
    cv2.line(frame, (0, image_center_y), (frame_width, image_center_y), (255, 255, 255), 1)

    if ids is None:
        cv2.putText(
            frame,
            "No ArUco marker detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return frame

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape((4, 2)).astype(int)

        top_left, top_right, bottom_right, bottom_left = pts

        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))

        error_x = center_x - image_center_x
        error_y = center_y - image_center_y

        # Draw marker center
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Draw line from image center to marker center
        cv2.line(frame, (image_center_x, image_center_y), (center_x, center_y), (0, 255, 255), 2)

        # Draw information
        info_1 = f"ID: {marker_id}"
        info_2 = f"Center: ({center_x}, {center_y})"
        info_3 = f"Error X: {error_x}px"

        cv2.putText(
            frame,
            info_1,
            (top_left[0], top_left[1] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            info_2,
            (top_left[0], top_left[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            info_3,
            (top_left[0], top_left[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Print to terminal for debugging
        print(f"Marker ID: {marker_id} | Center X: {center_x} | Error X: {error_x}")

    return frame


def main():
    detector = create_aruco_detector()

    # Use 0 for default webcam.
    # If your webcam does not open, try 1 or 2.
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try changing camera index from 0 to 1.")

    # Optional camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("ArUco detector started.")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        output_frame = draw_marker_info(frame, corners, ids)

        cv2.imshow("AkarBot ArUco Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()