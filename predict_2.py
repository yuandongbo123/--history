# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import time, datetime
import pdb, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import cv2
# import imagehash
from PIL import Image
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4') 

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import pydicom 
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# class MyDataset(Dataset):
#     """
#         construct the dataset
#     """
#     def __init__(self,images_path,images_label,transform=None,train=True):
# #         self.imgs = [os.path.join(img_path,"".join(path)) for path in images_path]
#         self.imgs  =  images_path
#         # if train dataset : get the appropriate label
#         if train:
#             self.train = True
#             self.labels = images_label
#         else:
#             self.train = False
        
#         # transform
#         self.transform = transform
        
#     def read_dicm(self,path, voi_lut=True, fix_monochrome=True):
#         dicom = pydicom.read_file(path)
#         # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
#         # "human-friendly" view
#         if voi_lut:
#             data = apply_voi_lut(dicom.pixel_array, dicom)
#         else:
#             data = dicom.pixel_array

#         # depending on this value, X-ray may look inverted - fix that:
#         if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#             data = np.amax(data) - data

#         data = data - np.min(data)
#         if np.max(data) != 0:
#             data = data / np.max(data)
#         data = (data * 255).astype(np.uint8)

#         return data
    
#     def crop(self,img):
#         # 以最长的一边为边长，把短的边补为一样长，做成正方形，避免resize时会改变比例
#         dowm = img.shape[0]
#         up = img.shape[1]
#         max1 = max(dowm, up)
#         dowm = (max1 - dowm) // 2
#         up = (max1 - up) // 2
#         dowm_zuo, dowm_you = dowm, dowm
#         up_zuo, up_you = up, up
#         if (max1 - img.shape[0]) % 2 != 0:
#             dowm_zuo = dowm_zuo + 1
#         if (max1 - img.shape[1]) % 2 != 0:
#             up_zuo = up_zuo + 1
#         matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),  # 向上填充n个维度，向下填充n个维度
#                                             (up_zuo, up_you),  # 向左填充n个维度，向右填充n个维度
#                                             (0, 0))  # 通道数不填充
#                             , mode="constant",  # 填充模式
#                             constant_values=(0, 0))
#         img = Image.fromarray(matrix_pad)
#         return img

#     def __getitem__(self,index):
#         image_path = self.imgs[index]
#         pil_img = self.read_dicm(image_path, voi_lut=True, fix_monochrome=True)
#         pil_img = self.crop(pil_img)
#         if self.transform:
#             transform = self.transform
#         else:
#             # if not define the transform:default resize the figure(224,224) and ToTensor
#             transform = transforms.Compose([
#                 transforms.Resize((224,224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
#                 ])
#         img_t = transform(pil_img)
#         if self.train:
#             image_label = self.labels[index]
#             return img_t,image_label
#         else:
#             return img_t 
    
#     def __len__(self):
#         return len(self.imgs)


class MyDataset(Dataset):
    """
        construct the dataset
    """
    def __init__(self,images_path,images_label,transform=None,train=True):
#         self.imgs = [os.path.join(img_path,"".join(path +'.png')) for path in images_path]
        self.imgs = images_path
        # if train dataset : get the appropriate label
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False

        self.transform = transform  
    
    def __getitem__(self,index):
#         image_path = self.imgs[index]  +'.png'
        img = Image.open(self.imgs[index])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform:
            transform = self.transform
        else:
            # if not define the transform:default resize the figure(224,224) and ToTensor
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        img_t = transform(img)
        if self.train:
            image_label = self.labels[index]
            return img_t,image_label
        else:
            return img_t 
    def __len__(self):
        return len(self.imgs)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
                
        model = models.resnet34(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 5)
        self.resnet = model
        
#         model = EfficientNet.from_pretrained('efficientnet-b4') 
#         model._fc = nn.Linear(1792, 2)
#         self.resnet = model
        
    def forward(self, img):        
        out = self.resnet(img)
        return out

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, input in enumerate(test_loader):
                input = input.cuda()
#                 target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta



args = config.args
test_data = [os.path.join('/home/tione/notebook/dcm2png_test', x ) for x in os.listdir('/home/tione/notebook/dcm2png_test') if 'test_' in x]
test_data = np.array(test_data)
img_path = '/home/tione/notebook/dcm2png_test'
test_pred = None
for model_path in ['resnet18_fold0.pt','resnet18_fold1.pt','resnet18_fold2.pt','resnet18_fold3.pt','resnet18_fold4.pt']:
    
    test_loader = torch.utils.data.DataLoader(
        MyDataset(test_data,
                transforms.Compose([
                            transforms.Resize((256, 256)),
                            # transforms.CenterCrop((450, 450)),
                            transforms.RandomHorizontalFlip(),
#                             transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            train=False
        ), batch_size=10, shuffle=False, num_workers=10, pin_memory=True
    )
        

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = MyNet().cuda()
        model = nn.DataParallel(model,device_ids=[0,1]) ###如果只有一个gpu把1去掉
        checkpoint = torch.load('./save/'+ model_path)
#         state_dict  = checkpoint['model']
        model.load_state_dict(checkpoint)
    # model = nn.DataParallel(model).cuda()
    if test_pred is None:
        test_pred = predict(test_loader, model, 5)
    else:
        test_pred += predict(test_loader, model, 5)
###提交csv

subpath = '/home/tione/notebook/taop-2021-result/01_results/1632483162'#.format(time.time())
if not os.path.exists(subpath):
    os.makedirs(subpath)
else:
    pass


test_csv = pd.DataFrame()
# test_csv['uuid'] = list(range(1, 2001))
test_csv['id_type'] = [1 for i in range(297)]
test_csv['id_project'] = [100001 for i in range(297)]
test_csv['id_patient'] = [x.split('.')[0] for x in os.listdir('/home/tione/notebook/dcm2png_test')]
test_csv['id_exam'] = [x.split('.')[0] for x in os.listdir('/home/tione/notebook/dcm2png_test')]
test_csv['id_series'] = test_csv.apply(lambda x:('') ,axis=1,result_type='expand')
test_csv['id_image'] = test_csv.apply(lambda x:(''),axis=1,result_type='expand')
test_csv['id_doctor'] = test_csv.apply(lambda x:(''),axis=1,result_type='expand')
test_csv['class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'] = np.argmax(test_pred, 1)
test_csv['class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'] = test_csv['class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'].map({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
test_csv.to_csv(subpath +'/' '01_results.csv', index=False)
