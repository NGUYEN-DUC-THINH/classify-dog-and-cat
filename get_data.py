import cv2
import os
from tqdm import tqdm
import numpy as np
import shutil




def move_data(org_data):
    for img in os.listdir(org_data):
        label = img.split('.')[0]
        if label =='cat':
            shutil.copy(org_data + '/'+ img , 'data/train/cat')
        elif label =='dog':
            shutil.copy(org_data + '/'+ img , 'data/train/dog')

dir = 'data/train'
org_data = 'data/org_data'
if os.path.exists(dir):
    shutil.rmtree(dir,ignore_errors=True)
os.mkdir(dir)
os.mkdir(dir+'/cat')
os.mkdir(dir+'/dog')
move_data(org_data)
                    


