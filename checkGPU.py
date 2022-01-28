# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:42:40 2022

@author: mark8
"""
import torch

def checkGPU(): 
    if torch.cuda.is_available(): 
        print(torch.cuda.is_available()) # bool, shows GPU availbale or not
    
        print(torch.cuda.device_count()) # GPU quantity
    
        print(torch.cuda.current_device()) # GPU index
    
        print(torch.cuda.get_device_name(0)) # GPU name

if __name__ == '__main__': 
    checkGPU()
