import os

import cv2
import py360convert as p3c


def convert_image_to_cubic(input_path, img_path, output_path):

    print("Processing image...")
    img = os.path.join(input_path, img_path)

    # Open and transform to a numpy array with shape [H, W, 3] using cv2 (convert to RGB)
    try:
        img = cv2.imread(img)
    except FileNotFoundError:
        print(f"File {img} not found")
        return

    # Project from equirectangular to cubic
    size = 1024
    front, right, back, left, top, bottom = p3c.e2c(
        img, face_w=size, mode="bilinear", cube_format="list"
    )

    # make directory, with panoid as name, to save them in
    folder = img_path.split(".")[0]
    directory = os.path.join(output_path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save cubemap faces
    cv2.imwrite(f"{directory}/front.png", front)
    cv2.imwrite(f"{directory}/right.png", right)
    cv2.imwrite(f"{directory}/back.png", back)
    cv2.imwrite(f"{directory}/left.png", left)
    # cv2.imwrite(f'{directory}/top.png', top)
    # cv2.imwrite(f'{directory}/bottom.png', bottom)


def main():
    print("We are in main")
    # Step 1: set the paths (for now, hardcode)
    # Set the path to the images
    input_path = "../../input_images"
    output_path = "../../output_images"

    # Step 2: convert the images
    for img_path in os.listdir(input_path):
        print("We are in the for loop")
        print(f"img_path: {img_path}")
        convert_image_to_cubic(input_path, img_path, output_path)

    # Step 3: convert the annotations


if __name__ == "__main__":
    print("Test1")
    main()
