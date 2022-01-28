import torch
import cv2
import numpy as np

# predict single img

def pred_img(): 
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    # yolov5m
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp12/weights/best.pt', force_reload=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp13/weights/best.pt')
    # print(model)
    # model.conf = 0.77
    
    img = cv2.imread('test_sample/homemade_canele_test2.jpg')
    results = model(img)
    results.print()
    print(results.xyxy)
    print(results.pandas().xyxy)
    cv2.imshow('output', np.squeeze(results.render()))
    cv2.waitKey(0)

if __name__ == '__main__': 
    pred_img()