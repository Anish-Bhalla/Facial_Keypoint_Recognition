# Facial KeyPoint Recognition


This repository implements a facial keypoint detection system using Convolutional Neural Network CNNs. It enables the identification of 15 crucial points on the face, including the 3 points for both the corners & a center point of the eyes, 2 for eyebrows (i.e. 3+2=5 points for one eye, so 5X2=10 points for both eyes), 1 for nose tip, and 4 points for the mouth. This can be used as a building block in several applications, such as:

- Creating Filters in social media apps

- Tracking faces in images and video

- Analysing facial expressions

- Detecting dysmorphic facial signs for medical diagnosis

- Biometrics/Face recognition

The input dataset consists of 7049 training images but with missing data of 11 facial key points out of 15 for approximately 4800 images. I created my CNN model in Tensorflow framework to train on clean data after dropping null values and used it to predict the missing labels. The Jupyter notebook demonstrates how I handled the missing data and used Data Augmentation to have 49343 images for training my model and finally, I also implemented Transfer learning using MobileNet architecture.

The models were scored using Root Mean Squared Error (RMSE) loss function and after submitting predictions on Kaggle, I achieved 1.45214 RMSE Score on Private Leaderboard at position 4 and 1.65826 RMSE Score on Public Leaderboard at position 5. 


### Sample Input Image with 15 Facial Key Points

![image](https://github.com/Anish-Bhalla/Facial_Keypoint_Recognition/assets/103365300/f6ef0ad4-30b0-4ff5-8c39-95afcd9b0d64)

### Credits

This project is an implementation of Kaggle Competition **Facial Keypoints Detection** and data is available at https://www.kaggle.com/competitions/facial-keypoints-detection/data
