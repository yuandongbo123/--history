IMAGE_SIZE = 240
SCALE = .96
NUM_IMAGES = 64
MRI_TYPE = "FLAIR"
##减小黑边
from  PIL import Image
def load_dicom_image(
    path,
    img_size=IMAGE_SIZE,
    scale=SCALE):
    '''
    This function allows you to load a DCIM type image 
    and apply preprocessing steps such as crop, resize 
    and denoising filter to it.
    ****************************************************
    PARAMETERS
    ****************************************************
    - path : String
        Path to the DCIM image file to load.
    - img_size : Integer
        Image size desired for resizing.
    - scale : Float
        Desired scale for the cropped image
    - prep : Bool
        True for a full preprocessing with
        denoising.
    '''
    # Load single image
    img = pydicom.read_file(path).pixel_array
    # Crop image
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    # Resize image
#     img = cv2.resize(img, (img_size, img_size))
    
    return img
    
    
def crop(img):
    # 以最长的一边为边长，把短的边补为一样长，做成正方形，避免resize时会改变比例
    dowm = img.shape[0]
    up = img.shape[1]
    max1 = max(dowm, up)
    dowm = (max1 - dowm) // 2
    up = (max1 - up) // 2
    dowm_zuo, dowm_you = dowm, dowm
    up_zuo, up_you = up, up
    if (max1 - img.shape[0]) % 2 != 0:
        dowm_zuo = dowm_zuo + 1
    if (max1 - img.shape[1]) % 2 != 0:
        up_zuo = up_zuo + 1
    matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),  # 向上填充n个维度，向下填充n个维度
                                        (up_zuo, up_you),  # 向左填充n个维度，向右填充n个维度
                                        (0,0))  # 通道数不填充
                        , mode="constant",  # 填充模式
                        constant_values=(0,0))
    img = Image.fromarray(matrix_pad)
    img = img.resize((224,224))
    return img 

import pandas as pd 
import numpy as np 
from tqdm import tqdm
import pydicom
from PIL import Image
import os
img_path = 'taop-2021/100001/To user/'
df = pd.read_csv('taop-2021/100001/To user/train1_data_info.csv')
imgs = [os.path.join(img_path,"".join(path)) for path in df['id_patient']]
if not os.path.exists('dcm2png_crop'):
    os.makedirs('dcm2png_crop')
else:
    pass

for i in tqdm(df['id_patient']):
    image_path ='taop-2021/100001/To user/'+ i  +'.dcm'
#     print(image_path)
    pil_img = load_dicom_image(image_path)
    pil_img = crop(pil_img)
    pil_img.save('dcm2png_crop/' + i + '.png')
