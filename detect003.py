# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:13:55 2022

@author: mark8
"""
import torch
import numpy as np
import cv2
import time

# predict webcam

def pred_webcam(): 
    prev_time = 0
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp13/weights/best.pt')
    # model.conf = 0.5
    capture = cv2.VideoCapture(0)
    while capture.isOpened(): 
        success, frame = capture.read()
        if not success: 
            print('Ignoring empty camera frame')
            continue
        frame = cv2.resize(frame, (640, 640))
        results = model(frame)
        output_img = np.squeeze(results.render())
        cv2.putText(output_img, f'FPS: {int(1 / (time.time()- prev_time))}', (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        prev_time = time.time()
        cv2.imshow('output', output_img)
        print(results.pandas().xyxy)
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    pred_webcam()