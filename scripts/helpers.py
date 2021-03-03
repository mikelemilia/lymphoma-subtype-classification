import cv2
import os


def load_images(path):
    images = []
    labels = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            image = cv2.imread(filename)
            label = os.path.basename(os.path.dirname(filename))

            if image is not None:
                images.append(image)
                labels.append(label)

    return images, labels
