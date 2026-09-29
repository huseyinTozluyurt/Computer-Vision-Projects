import cv2
import os


def main():
    output_dir = "markers"
    os.makedirs(output_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    marker_ids = [0, 1, 5, 6, 11, 12, 20]

    marker_size_pixels = 600

    for marker_id in marker_ids:
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            marker_size_pixels
        )

        output_path = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(output_path, marker_image)

        print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()