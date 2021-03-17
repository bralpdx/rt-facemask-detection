"""
Model based off of the following sources:
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
https://www.mygreatlearning.com/blog/real-time-face-detection/#sh2
"""
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from prepare_data import IMG_SIZE
import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pickle.load(open("x.pickle", "rb"))  # NumPy Arrays of images
labels = pickle.load(open("y.pickle", "rb"))  # Image Labels. 0 indicates mask, 1 indicates no mask.

# Convert to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Implements pre-trained model as base layer. Pre-trained on imagenet.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output  # Initial Convolution Layer
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # Average Pooling Layer
headModel = Flatten(name="flatten")(headModel)  # Flattens pooling layer
headModel = Dense(128, activation="relu")(headModel)  # Fully connected layer
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)  # Fully connected layer

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20,
                                                  stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# Plots model values
def show_plot(epochs, H, message):
    N = epochs
    x = np.arange(0, N)

    plt.style.use("ggplot")
    plt.figure()

    y = cat_to_np("loss")
    plt.plot(x, y, label="train_loss")
    y = cat_to_np("val_loss")
    plt.plot(x, y, label="val_loss")
    y = cat_to_np("accuracy")
    plt.plot(x, y, label="train_acc")
    y = cat_to_np("val_accuracy")
    plt.plot(x, y, label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.suptitle(message)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


# Converts to np array via category
def cat_to_np(cat):
    return H.history[cat]


if __name__ == "__main__":
    INIT_LR = 1e-4  # Learning rate
    EPOCHS = 20
    BS = 32
    STEPS_PER_EPOCH = len(trainX) // BS
    STEPS_VALIDATION = len(testX) // BS

    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=(testX, testY),
        validation_steps=STEPS_VALIDATION,
        epochs=EPOCHS)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

    model.save("mask_recog_ver3.h5")
    message = "New dataset"
    show_plot(EPOCHS, H, message)
