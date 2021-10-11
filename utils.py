#########################ben 的图片预处理pipline#########################

class MyDataset(Data.Dataset):
    """
        construct the dataset
    """
    def __init__(self,images_path,images_label,transform=None,train=True):
        self.imgs = [os.path.join(img_path,"".join(path +'.png')) for path in images_path]
        # if train dataset : get the appropriate label
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False

        self.transform = transform 
    def load_dicom_image(self,
        img,
        img_size=512,
        scale=0.98):
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
    def crop_image(self,img, tol=7):
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
        
    def circle_crop(self,img):
        img = self.crop_image(img)

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
        img = self.crop_image(img)

        return img    
    def __getitem__(self,index):
#         image_path = self.imgs[index]  +'.png'
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.circle_crop(img)
        img = self.load_dicom_image(img)
        # RGB为彩色图片，L为灰度图片
        img = Image.fromarray(img)
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
        
        
        
        
        
#################################################自适应剪切图片#########################################################
class MyDataset(Data.Dataset):
    """
        construct the dataset
    """
    def __init__(self,images_path,images_label,transform=None,train=True):
        self.imgs = [os.path.join(img_path,"".join(path)) for path in images_path]
        # if train dataset : get the appropriate label
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False
        
        # transform
        self.transform = transform
        
    def read_dicm(self,path, voi_lut=True, fix_monochrome=True):
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
    
     def crop(self, img):
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
                                            (0, 0))  # 通道数不填充
                            , mode="constant",  # 填充模式
                            constant_values=(0, 0))
        img = Image.fromarray(matrix_pad)
        return img

    def __getitem__(self,index):
        image_path = self.imgs[index]  +'.dcm'
        pil_img = self.read_dicm(image_path, voi_lut=True, fix_monochrome=True)
        pil_img = self.crop(pil_img)
        if self.transform:
            transform = self.transform
        else:
            # if not define the transform:default resize the figure(224,224) and ToTensor
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                ])
        img_t = transform(pil_img)
        if self.train:
            image_label = self.labels[index]
            return img_t,image_label
        else:
            return img_t 
    
    def __len__(self):
        return len(self.imgs)
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        
        
###############################随机方块噪声###########################################
class RandomErase(object):
    def __init__(self, prob, sl, sh, r):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r = r

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img

        while True:
            area = random.uniform(self.sl, self.sh) * img.size[0] * img.size[1]
            ratio = random.uniform(self.r, 1/self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.size[0] and w < img.size[1]:
                x = random.randint(0, img.size[0] - h)
                y = random.randint(0, img.size[1] - w)
                img = np.array(img)
                if len(img.shape) == 3:
                    for c in range(img.shape[2]):
                        img[x:x+h, y:y+w, c] = random.uniform(0, 1)
                else:
                    img[x:x+h, y:y+w] = random.uniform(0, 1)
                img = Image.fromarray(img)

                return img
        
##########################加载例子#############################################
train_transform = transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224 x 224的新图像
    transforms.RandomRotation(degrees=60, expand=False),
    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#     transforms.RandomAffine(
#                             degrees=(-90, 90),
#                             scale=(0.8889, 1.0),
#                             shear=(-36, 36)),
# #     transforms.RandomAffine(10),
    # 随机更改亮度，对比度和饱和度
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ColorJitter(contrast=(0.9, 1.1)),
    添加随机噪声 随机进行遮挡
    RandomErase(
            prob=0.5,
            sl=0.01,
            sh=0.1,
            r=0.1),
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

val_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

# train_dataset,label_map,label_inv_map = load_data(train_transform=train_transform,test_transform=None)
train_dataset = MyDataset(train_data['id_patient'],train_data['label'],transform=train_transform,train=True)
train_loader = torch.utils.data.DataLoader(
    MyDataset(train_data['id_patient'],
              np.array(train_data['label'].astype('int64')),
              train_dataset,
              train=True),
            batch_size=10, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
print(len(train_loader))

############################可视化显示#################################################################
#没有显示 ，在运行一次
import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.figure(figsize=(16,16))
for i in range(1,6,2):
    img, label = train_dataset[10]
    img = np.transpose(img, (1,2,0))
    img = img*0.5 + 0.5
    plt.subplot(3,2,i),plt.imshow(img,'gray'),plt.title('{}'.format(i)),plt.xticks([]),plt.yticks([])
#     plt.subplot(3,2,i+1),plt.imshow(label,'gray'),plt.title('label'),plt.xticks([]),plt.yticks([])
    plt.show
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
