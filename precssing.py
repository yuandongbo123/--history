###################################预处理可视化#################################3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm
import cv2
from PIL import Image
df = pd.read_csv('/home/tione/notebook/taop-2021/100001/To user/train1_data_info.csv')
train_df = df[:2400]
valid_df = df[2400:]
print(train_df.shape)
print(valid_df.shape)
train_df.head()


def display_samples(df, columns=5, rows=5):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_patient']
        image_id = df.loc[i,'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)']
        img = cv2.imread(f'/home/tione/notebook/dcm2png/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)




def crop_image(img, tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
            
        return img
        
        
        
def display_samples(df, columns=5, rows=5):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_patient']
        image_id = df.loc[i,'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)']
        img = cv2.imread(f'/home/tione/notebook/dcm2png/{image_path}.png')
        img = crop_image(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


def circle_crop(img):
    img = crop_image(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = width//2
    y = height//2
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image(img)

    return img
 def display_samples(df, columns=5, rows=5):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_patient']
        image_id = df.loc[i,'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)']
        img = cv2.imread(f'/home/tione/notebook/dcm2png/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = circle_crop(img)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)

def load_dicom_image(
    img,
    img_size=512,
    scale=0.8):
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
#     img = pydicom.read_file(path).pixel_array
    # Crop image
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    # Resize image
#     img = cv2.resize(img, (img_size, img_size))
    
    return img
    
    
    
def display_samples(df, columns=5, rows=5):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_patient']
        image_id = df.loc[i,'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)']
        img = cv2.imread(f'/home/tione/notebook/dcm2png/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = circle_crop(img)
        img = load_dicom_image(img,img_size=512,scale=0.96)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)
