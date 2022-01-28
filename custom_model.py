# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 00:08:57 2022

@author: mark8
"""

import torch
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt', force_reload=True)
capture = cv2.VideoCapture(0)

while capture.isOpened(): 
    success, frame = capture.read()
    if not success: 
        print('Ignoring empty camera frame...')
        continue
    frame = cv2.resize(frame, (800,480))
    result = model(frame)
    cv2.imshow('output', np.squeeze(result.render()))
    if cv2.waitKey(100) & 0xFF == 27: 
        break

capture.release()    
cv2.destroyAllWindows()