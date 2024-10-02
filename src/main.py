from model import load_image, segment_image, setup_model
from input import (display_image, display_segmented_object, load_background,
                   save_image, stitch_object)


def main(image_path, background_path):
    image = load_image(image_path)
    background = load_background(image.shape, background_path)
    predictor = setup_model()
    masks = segment_image(predictor, image)

    for i, mask in enumerate(masks):
        object_pixels = image.copy()
        object_pixels[mask == 0] = 0

        if display_segmented_object(object_pixels, i):
            stitch_object(background, mask, image)

    display_image(background, "Stitched Image on Background")

    save_image(background, "stitched_image_on_background.png")


if __name__ == "__main__":
    image_path = "image.jpg"
    background_path = "park.jpg"
    main(image_path, background_path)
