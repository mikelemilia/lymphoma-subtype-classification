import models
import helpers
import cv2
import argparse

if __name__ == '__main__':
    images, labels = helpers.load_images('data')
    for i in range(0, len(images)):
        resized = cv2.resize(images[i], (500, 500))
        cv2.imshow(labels[i], resized)
        cv2.waitKey(2)
    print(len(images))
    print(len(labels))

    # c = models.conv()
