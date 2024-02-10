import pandas as pd
import time
import numpy as np
from keras.models import load_model
import mediapipe as mp
import cv2
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model file relative to the current directory
model_file = os.path.join(
    current_dir, '../Models/Model_Sign_Language_MNIST.h5')
# Load the trained model
model = load_model(model_file)
# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the default camera
cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

# Define the alphabet letters
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Process the frame with MediaPipe Hands
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_landmarks = result.multi_hand_landmarks

    # Draw hand landmarks and rectangles around detected hands
    if hand_landmarks:
        for handLMs in hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

            # Calculate bounding box around the hand
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Draw rectangle around the hand
            cv2.rectangle(frame, (x_min - 20, y_min - 20),
                          (x_max + 20, y_max + 20), (0, 255, 0), 2)

            # Crop and preprocess the hand region for gesture recognition
            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min -
                                          20:y_max + 20, x_min - 20:x_max + 20]
            analysisframe = cv2.resize(analysisframe, (28, 28))
            analysisframe = analysisframe / 255
            analysisframe = np.expand_dims(analysisframe, axis=0)
            analysisframe = np.expand_dims(analysisframe, axis=-1)

            # Make prediction using the model
            prediction = model.predict(analysisframe)
            predarray = np.array(prediction[0])
            letter_prediction_dict = {
                letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    print("Predicted Character 1: ", key)
                    print('Confidence 1: ', 100 * value)
                elif value == high2:
                    print("Predicted Character 2: ", key)
                    print('Confidence 2: ', 100 * value)
                elif value == high3:
                    print("Predicted Character 3: ", key)
                    print('Confidence 3: ', 100 * value)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for key press
    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE pressed
        time.sleep(5)


# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
