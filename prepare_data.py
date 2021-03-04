"""
This script creates a training dataset, from the image data
located in the mask-data subdirectories.

Script expects the presence of a mask-data directory, and
a with_mask, and without_mask subdirectories.
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
import pickle

BASEDIR = "mask-data"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 224


# Returns a training data-set
def create_training_data():
    training_data = list()
    for cat in CATEGORIES:
        path = os.path.join(BASEDIR, cat)
        if not os.path.isdir(path):
            print("Directory {path} not found. Check to make sure it exists.".format(path=path))
            exit()
        class_num = CATEGORIES.index(cat)
        for file in os.listdir(path):
            # Converts image to grayscale, and reads it into a numpy array
            img = load_img(os.path.join(path, file), target_size=(224, 224))
            try:
                img = img_to_array(img)
            except Exception:
                print(file)
            img = preprocess_input(img)
            training_data.append([img, CATEGORIES[class_num]])

    return training_data


# Saves image data (numpy array, and label)
# in x.pickle and y.pickle respectively.
def save_to_pickle(x, y):
    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    print("[INFO] Preparing data...")
    data = create_training_data()
    random.shuffle(data)
    x = list()
    y = list()

    for features, label in data:
        x.append(features)
        y.append(label)

    x = np.array(x)

    print("[INFO] Saving data...")
    save_to_pickle(x, y)
    print("[INFO] Finished.")

