# Yolov5 custom training
![title_dessert](https://user-images.githubusercontent.com/51041343/152417402-0476e02b-50eb-49c6-afe4-ecc8058bbf90.PNG)

### Anaconda Enviroment

- CUDA_11.6.0
- cudnn_8.3.2.44
- GTX 1070 (8192MB)
- Python_3.7.11
- Numpy_1.21.5
- Opencv_4.5.62
- Pytorch_1.10.1+cu113
- Tensorflow-gpu_2.7.0
- Pyqt5_5.9

### How to train

1. put training data inside `datasets/images` folder

2. set `labelImg/data/predefined_classes.txt` to corresponding classes

3. cd to labelImg folder`$python labelImg.py`, remember to set output folder `datasets/labels`

4. edit `yolov5/dataset.yaml` for your needs
```python
# dataset.yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/dessert  # dataset root dir
train: images  # train images (relative to 'path') 128 images
val: images  # val images (relative to 'path') 128 images
test:  # test images (optional)
# Classes
nc: 8  # number of classes
names: [ 'canele', 'cremeBrulee', 'croissant', 'lemonMeringuePie', 'macaron', 'madeleine', 'millefeuille', 'souffle' ]  # class names
```
5. run `$python train.py --img 640 --batch 1 --epochs 3 --data dataset.yaml --weights yolov5s.pt` you can edit img_size, batch_size, pre-trained model, epochs... to fit your requirement

6. after training, all the results are in `yolov5/runs/train` folder and the trained model at `yolov5\runs\train\expx\weights`

7. to use (load) custom model for detecting images
```python
# detect000.py
...
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp?/weights/best.pt', force_reload=True)
...
```

### Frequently error and solution

#### - Error while using command pyrcc5 –o libs/resources.py…
> File"C:\Users\mark8\anaconda3\envs\YOLO\lib\sitepackages\PyQt5\pyrcc_main.py", line 23, in <module>
from .pyrcc import *
ImportError: DLL load failed: The specified procedure could not be found.

1. copy `anaconda3/envs/'env_name'/python3.dll` to `anaconda3/envs/'env_name'/ Scripts`

2. `$pip install --user pyqt5==5.9`

#### - CV2: "[ WARN:0] terminating async callback" when attempting to take a picture
> !_src.empty() in function 'cvtColor' error

1.  `cmd` `setx OPENCV_VIDEOIO_PRIORITY_MSMF 0`

2. 
```python
# takepic.py 
...
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
...
```

#### - Error while using train.py
> TypeError: ‘NoneType‘…brokenpipe…

1. check GPU maximum batch size

2. `$python train.py --img 640 --batch 1 --epochs 3 --data dataset.yaml --weights yolov5s.pt` by using this command to check with batch size == 1 and pre-trained model yolov5s

> OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.OMP:

1. in train.py
```
...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
...
```

### Improvement
![improve](https://user-images.githubusercontent.com/51041343/152417879-8c66fd98-4d24-4aa8-b164-d6ccbda28203.PNG)

### Reference

- [Anaconda + Tensorflow-GPU+ CUDA + cuDNN env establishment](https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-windows-%E6%90%AD%E5%BB%BAtensorflow-gpu-%E7%92%B0%E5%A2%83-anaconda-tensorflow-gpu-cuda-cudnn-a047c0f275f4)
- [Yolov5](https://github.com/ultralytics/yolov5)
- [Yolov5 Pytorch](https://pytorch.org/hub/ultralytics_yolov5/)
- [labelImg](https://github.com/tzutalin/labelImg)
- [Yolov5 Tips for Best Training Results](https://docs.ultralytics.com/tutorials/training-tips-best-results/)
- [Yolov5 Pytorch Hub parameter setting](https://docs.ultralytics.com/tutorials/pytorch-hub/)




