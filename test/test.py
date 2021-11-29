#2021/11/2 13:32
import torch
import numpy as np
import torchvision as tv
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image

img=cv2.imread('D:\Code\Autopilot\data\IMG\center_2021_11_01_13_15_09_072.jpg')
print(img.shape)

# print(type(img))
# img = torch.from_numpy(img)
# print(type(img))
# print(img.shape)
# img = img.float()
# img = torch.reshape(img, ( 3, 320, 160))
img = np.asarray(img).transpose(-1, 1, 0)
print(img.shape)
# img=np.transpose(img,(0,1,2))
img=np.transpose(img,(-1,1,0))
print(img.shape)
# 显示图片
plt.imshow(img)
plt.show()

# transform = transforms.Compose(
#     [
#         transforms.ToTensor()
#     ]
# )
# train_set = tv.datasets.ImageFolder(root='./', transform=transform)
# # data_loader = DataLoader(dataset=train_set)
#
# to_pil_image = transforms.ToPILImage()
#
# for image, label in data_loader:
#     # 方法1：Image.show()
#     # transforms.ToPILImage()中有一句
#     # npimg = np.transpose(pic.numpy(), (1, 2, 0))
#     # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
#     img = to_pil_image(image[0])
#     img.show()
#
#     # 方法2：plt.imshow(ndarray)
#     img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
#     img = img.numpy()  # FloatTensor转为ndarray
#     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
#
#     # 显示图片
#     plt.imshow(img)
#     plt.show()