# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:56:30 2023

@author: jadel
"""
#citation for thin plate splicing code AlanLuSun https://github.com/AlanLuSun/TPS-Warp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")
show_camera()




def WarpImage_TPS(source,target,img):
	tps = cv2.createThinPlateSplineShapeTransformer()

	source=source.reshape(-1,len(source),2)
	target=target.reshape(-1,len(target),2)

	matches=list()
	for i in range(0,len(source[0])):

		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target, source, matches)  # note it is target --> source

	new_img = tps.warpImage(img)

	# get the warp kps in for source and target
	tps.estimateTransformation(source, target, matches)  # note it is source --> target
	# there is a bug here, applyTransformation must receive np.float32 data type
	f32_pts = np.zeros(source.shape, dtype=np.float32)
	f32_pts[:] = source[:]
	transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
	f32_pts = np.zeros(target.shape, dtype=np.float32)
	f32_pts[:] = target[:]
	transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

	return new_img, new_pts1, new_pts2

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
	rand_p=np.random.choice(x.size,num_points,replace=False)
	movingPoints=np.zeros((1,num_points,2),dtype='float32')
	fixedPoints=np.zeros((1,num_points,2),dtype='float32')

	movingPoints[:,:,0]=x[rand_p]
	movingPoints[:,:,1]=y[rand_p]
	fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
	fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

	tps=cv2.createThinPlateSplineShapeTransformer()
	good_matches=[cv2.DMatch(i,i,0) for i in range(num_points)]
	tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

	imh,imw=imshape
	x,y=np.meshgrid(np.arange(imw),np.arange(imh))
	x,y=x.astype('float32'),y.astype('float32')
	# there is a bug here, applyTransformation must receive np.float32 data type
	newxy=tps.applyTransformation(np.float32((x.ravel(),y.ravel())))[1]
	newxy=newxy.reshape([imh,imw,2])

	if offsetMatrix:
		return newxy,newxy-np.dstack((x,y))
	else:
		return newxy

def main():
  # the correspondences need at least four points
  shirts = np.array([[321,75],[320,281],[469,91],[177,87],[321,75],[320,281],[469,91],[177,87],[321,75],[320,281],[469,91],[177,87],[321,75],[320,281],[469,91],[177,87]])
  selection = input("Hi, thank you for using FitCheck. Please select r for red, b for blue, p for plad")
  if selection == "r":
	 source = shirts[0]
  if selection == "b":
	source = shirts[1]
  if selection == "p":
  	source = shirts[2]

  Zp = source # (x, y) in each row
  Zs = np.array(destination)
  im = cv2.imread('buttonupblackbackground.png')
  r = 6

  # draw parallel grids
  #for y in range(0, im.shape[0], 10):
  #		im[y, :, :] = 255
  #for x in range(0, im.shape[1], 10):
  #		im[:, x, :] = 255

  new_im, new_pts1, new_pts2 = WarpImage_TPS(Zp, Zs, im)
  new_pts1, new_pts2 = new_pts1.squeeze(), new_pts2.squeeze()
  print(new_pts1, new_pts2)

  # new_xy = thin_plate_transform(x=Zp[:, 0], y=Zp[:, 1], offw=3, offh=2, imshape=im.shape[0:2], num_points=4)

  #for p in Zp:
  #	cv2.circle(im, (p[0], p[1]), r, [0, 0, 255])
  #for p in Zs:
  #	cv2.circle(im, (p[0], p[1]), r, [255, 0, 0])
  #cv2.imshow('w', im)
  #cv2.waitKey(500)


  #for p in Zs:
  #	cv2.circle(new_im, (p[0], p[1]), r, [255, 0, 0])
  #for p in new_pts1:
  #	cv2.circle(new_im, (int(p[0]), int(p[1])), 3, [0, 0, 255])
  cv2.imread(new_im)
  cv2.flip(new_im)
  cv2.imshow('w2', new_im)
  cv2.waitKey(0)
  cv2.destroyWindow(new_im)
