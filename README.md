# Real Time Facemask Detection
## Using OpenCV, and Tensorflow
### How to run
---
First, make sure you have the required libraries installed (shown in requirements.txt).

If not installed, it is recommended to do this in a virtual environment.

Once in the desired environment run an install command, for example:
```
pip install -r requirements.txt
```

To run the program, run the following command:
```
python face_detection.py
```
A window should then open displaying real time video from your default video capture device.

The program then draws rectangles around all faces detected in each frame.

A model prediction is then made for each face in the frame, and the output of the prediction is displayed on the frame.

I.E. Red rectangle for 'No Mask Detected', and a Blue rectangle for 'Mask Detected', along with a confidence percentage.

### About
---
This application based off of the following model:
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

Which takes advantage of a base model that has been pretrained on ImageNet, on which our Convolutional Neural Network was built.

I processed the images included in the following dataset, to train the model on:
https://www.kaggle.com/sumansid/facemask-dataset

Currently, the model performs poorly on individuals wearing glasses, which I suspect is due to a lack of sufficient data included in the dataset that was used.
