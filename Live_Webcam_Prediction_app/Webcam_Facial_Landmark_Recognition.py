import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
# Load our trained keypoint detection model
cwd = os.getcwd()
model_path = os.path.join(cwd,"mobilenet_model")
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

cap = cv2.VideoCapture(0)
detector = MTCNN()
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame,(96,96))
    frame = np.expand_dims(frame,axis=-1)
    frame = np.repeat(frame,3,axis=2)
    return frame
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Preprocess frame (e.g., resize, normalize)
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        facial_image = frame[y:y+height,x:x+width,:]
        processed_frame = preprocess_frame(facial_image)
        
        # Make predictions using the CNN model
        predictions = model(tf.constant(np.expand_dims(processed_frame,axis=0)))
        predictions = np.array(predictions['output_0'])

        x_in_96,y_in_96 = predictions[0,::2],predictions[0,1::2]
        x_in_96.shape,y_in_96.shape

        x_in_facial_image = (x_in_96 / 96)*facial_image.shape[1]
        y_in_facial_image = (y_in_96 / 96)*facial_image.shape[0]

        x_in_frame = x_in_facial_image + x
        y_in_frame = y_in_facial_image + y

        # Draw facial landmarks on the frame
        for i in range(15):
            cv2.circle(frame, (int(x_in_frame[i]), int(y_in_frame[i])), 3, (0, 0, 255),-1)
        
        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Facial Landmarks', frame)

             
        # Exit loop if 'qq' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Ask user to save the frame
save_frame = input("Do you want to save the last frame as image? (y/n): ")
if save_frame.lower() == 'y':
    # Save frame as JPG image
    cv2.imwrite(f'predictions.jpg', frame)

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()