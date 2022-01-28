# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:51:52 2022

@author: mark8
"""
import cv2
from time import strftime
import os

def take_pic(): 
    # labels = []
    # f = open('data/labels/mask.txt', 'r')
    
    # for line in f.readlines(): 
    #     labels.append(line)
    # f.close()
    labels = ['mask', 'nomask']
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while capture.isOpened(): 
        success, frame = capture.read()
        cv2.imshow('take_pic', frame)
        
        # 每0.1s LOOP
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27: # ESC
            break
        elif key == ord('0') or key == ord('1'): 
            # ord('0') == 48, 故按0代表拍一張mask；按1代表拍一張nomask
            print(key - 48)
            systime = strftime('%y%m%d%H%M%S')
            # 將mask和nomask加入檔案名稱中以便區隔
            imgdir = os.path.join('data/images', labels[key - 48] + '-'  + systime + '.jpg')
            cv2.imwrite(imgdir, frame)
    
    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__': 
    take_pic()