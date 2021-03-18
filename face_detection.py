from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os

# Initializes Video Input to Webcam
video_input = cv2.VideoCapture(0)
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

face_casc = cv2.CascadeClassifier(cascPath)

# model = load_model("mask_recog_pt.h5")  # Pre-trained model
IMG_SIZE = 224

model = load_model("mask_recog_ver4.h5")  # Our custom model

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)

# Check if the webcam is opened correctly
if not video_input.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = video_input.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.rectangle(frame, (20, 40), (200, -40), black, -1)
    cv2.putText(frame, "ESC to exit...", (25, 25), font, 0.75, white, 2, cv2.LINE_4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_casc.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(60, 60),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

    faces = []  # Faces detected by cascade model
    predictions = []  # Predictions made my mask model
    for (x, y, w, h) in detected_faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
        facial_frame = frame[y:y+h, x:x+w]
        facial_frame = cv2.cvtColor(facial_frame, cv2.COLOR_BGR2RGB)
        facial_frame = cv2.resize(facial_frame, (IMG_SIZE, IMG_SIZE))
        facial_frame = img_to_array(facial_frame)
        facial_frame = np.expand_dims(facial_frame, axis=0)
        facial_frame = preprocess_input(facial_frame)
        faces.append(facial_frame)

        if len(faces) > 0:
            # Prevents exception when faulty input is detected
            try:
                predictions = model.predict(faces)
            except Exception:
                pass

        for p in predictions:
            (mask, noMask) = p

        if mask > noMask:
            label = "Mask"
        else:
            label = "No Mask"
        if label == "Mask":
            frame_color = (255, 150, 0)
        else:
            frame_color = (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, noMask) * 100)  # Shows percentage of prediction
        cv2.rectangle(frame, (x, y), (x + w, y - 25), black, -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), frame_color, 2)

    # Display the resulting frame
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    # Quits on hitting the 'ESC' Key
    if c == 27:
        break

video_input.release()
cv2.destroyAllWindows()
