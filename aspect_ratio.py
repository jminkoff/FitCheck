# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:27:02 2023

@author: jadel
"""

import pygame
from sys import argv
from win32api import GetSystemMetrics
pygame.init()
pygame.display.init()

img=pygame.image.load(argv[-1]) #Get image file opened with it

width=GetSystemMetrics(0)*0.9   #Window size a bit smaller than monoitor size
height=width*img.get_height()/img.get_width()  # keep ratio

if height > GetSystemMetrics(1)*0.8:  # too tall for screen
    width = width * (GetSystemMetrics(1)*0.8)/height  # reduce width to keep ratio 
    height = GetSystemMetrics(1)*0.8  # max height

img=pygame.transform.scale(img, (int(width), int(height)))  #Scales image to window size
Main=pygame.display.set_mode((int(width), int(height)))
pygame.display.set_caption(str(argv[-1].split("\\")[-1]))   #Set window title as filename
imgrect=img.get_rect()

Main.blit(img, imgrect)
pygame.display.update()
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            exit()