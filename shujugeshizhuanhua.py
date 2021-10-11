import torch.utils.data as Data 
from torchvision import transforms
import torchvision
from PIL import Image 
import os 
# import d2l.torch as d2l 
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# 对label进行编码，并将映射表保存下来
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import torch 
from torch import nn 
from sklearn.model_selection import train_test_split,KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import ttach as tta 

def read_dicm(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
    
    
    
 img_path = 'taop-2021/100001/To user/'
# df = pd.read_csv('taop-2021/100001/To user/train1_data_info.csv')
imgs = [os.path.join(img_path,"".join(path)) for path in os.listdir(img_path)]
if not os.path.exists('dcm2png_test'):
    os.makedirs('dcm2png_test')
else:
    pass

for i in tqdm(imgs):
    path = i.split('/')[-1].split('.')[0]
    if 'test_' in i:
#         path2 = os.path.join('dcm2png_test',''.join(i.split('/')[-1].split('.')[0]+'.png'))
        pil_img = read_dicm(i, voi_lut=True, fix_monochrome=True)
        pil_img = Image.fromarray(pil_img)
        pil_img.save(os.path.join('dcm2png_test',''.join(i.split('/')[-1].split('.')[0]+'.png')))
    else:
        pass
#     pil_img = read_dicm(image_path, voi_lut=True, fix_monochrome=True)
#     pil_img = Image.fromarray(pil_img)
#     pil_img.resize((224,224))
#     pil_img.save('dcm2png/' + i + '.png')
    
