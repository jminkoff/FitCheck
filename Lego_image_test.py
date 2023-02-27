# -*- coding: utf-8 -*-
"""

@author: dabuk
"""
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

# Loads the specified image we want, the LEGO figure
img = cv2.imread("C:/Users/dabuk/Documents/EE434/input/jurassic-world/0001/009.jpg")

# Intialization of the MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Convert image to RGB for MediaPipe processing
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect pose landmarks using MediaPipe Pose
results = pose.process(img_rgb)

# Draw pose landmarks on the image
if results.pose_landmarks:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display image with pose estimation overlayed
cv2.imshow("LEGO figure with Pose Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
