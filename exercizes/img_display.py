import cv2

image = cv2.imread('DATA/chair.jpeg')

def display_image_shape(image, name):

    width, height, channels = image.shape
    print("******************************************************")
    print("*    " + str(name))
    print("*    Width: " + str(width))
    print("*    Height: " + str(height))
    print("*    Channels: " + str(channels))
    print("******************************************************")
    print("\n\n\n")


display_image_shape(image,"Original Image")

cv2.imshow("Original Image", image)
resized_img = cv2.resize(image,(800,640))

display_image_shape(resized_img, "Resized Image")

cv2.imshow("Resized Image",resized_img)

cv2.waitKey(0)

cv2.destroyAllWindows()