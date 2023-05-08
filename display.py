#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:03:08 2023

@author: fitcheck
"""

import cv2
import sys

from PIL import Image, ImageTk

def exit_program(event):
    if event.keysym == 'Escape':
        event.widget.destroy()


import PIL as Image

def main():
  # the correspondences need at least four points
 
  selection = input("Hi, thank you for using FitCheck. Please select r for red, b for blue, p for plad")
  if selection == "r":
    
    def showPIL(pilImage):
        root = tkinter.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.overrideredirect(1)
        root.geometry("%dx%d+0+0" % (w, h))
        root.focus_set()    
        root.bind("<Key>", exit_program) # bind exit_program function to any key event
        canvas = tkinter.Canvas(root,width=w,height=h)
        canvas.pack()
        canvas.configure(background='black')
        imgWidth, imgHeight = pilImage.size
        if imgWidth > w or imgHeight > h:
            ratio = min(w/imgWidth, h/imgHeight)
            imgWidth = int(imgWidth*ratio)
            imgHeight = int(imgHeight*ratio)
            pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(pilImage)
            imagesprite = canvas.create_image(w/2,h/2,image=image)
            root.mainloop()

        pilImage = Image.open("/home/fitcheck/FitCheck/final_mappingnolines.png")
        showPIL(pilImage)
         
  if selection == "b":
    
    imageObject = Image.open("finalmappingblue.png")
    
    
    hori_flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)
    Vert_flippedImage = imageObject.transpose(Image.FLIP_TOP_BOTTOM)
    Vert_flippedImage.show("finalmappingblue.png")
             
    
  if selection == "p":
   
    imageObject = Image.open("final mapping plad.png")
    
    hori_flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)
    Vert_flippedImage = imageObject.transpose(Image.FLIP_TOP_BOTTOM)
    Vert_flippedImage.show("final mapping plad.png")
             

  

   
    