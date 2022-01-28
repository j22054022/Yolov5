# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 22:29:41 2022

@author: mark8
"""

# cctv will automatically disconnect after 10~15s

import torch
import numpy as np
import cv2
import time

def pred_cctv(): 
    prev_time = 0
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    capture = cv2.VideoCapture('https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=10000&0.35507177477760843')
    while capture.isOpened(): 
        success, frame = capture.read()
        if not success: 
            print('Ignoring empty camera frame')
            continue
        frame = cv2.resize(frame, (960, 540))
        results = model(frame)
        output_img = np.squeeze(results.render())
        cv2.putText(output_img, f'FPS: {int(1 / (time.time()- prev_time))}', (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        prev_time = time.time()
        cv2.imshow('output', output_img)
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__': 
    pred_cctv()