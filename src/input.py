import cv2
import numpy as np


def display_segmented_object(object_pixels, index):
    cv2.imshow(f"Segmented Object {index}", object_pixels)
    print(
        f"PRESS 's' to save object {index}, or press any other key to skip."
    )
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord("s")


def stitch_object(stitched_image, mask, original_image):
    stitched_image[mask == 1] = original_image[mask == 1]


def create_empty_canvas(image_shape):
    return np.zeros(image_shape, dtype=np.uint8)


def save_image(image, file_name):
    cv2.imwrite(file_name, image)


def display_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_background(image_shape, background_path):
    background = cv2.imread(background_path)
    if background.shape != image_shape:
        background = cv2.resize(background, (image_shape[1], image_shape[0]))
    return background
