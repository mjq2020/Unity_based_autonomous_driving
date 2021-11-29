#2021/10/29 16:46
from cv2 import cv2

def filpimgs(datadc:dict):
    '''对图像进行镜像翻转，增加数据集数量'''
    for i in datadc.keys():
        print(i)
        if datadc[i]==0:continue
        if datadc[i]<0:
            img_font=cv2.imread(i)
            img_end=cv2.flip(img_font,1)
            cv2.imshow('font',img_font)
            cv2.imshow('end',img_end)
            print(datadc[i])
            cv2.waitKey(0)
        else:
            img_font=cv2.imread(i)
            img_end=cv2.flip(img_font,1)
            cv2.imshow('font',img_font)
            cv2.imshow('end',img_end)
            print(datadc[i])
            cv2.waitKey(0)

def flipimg(img):
    '''对图像进行水平翻转'''
    return cv2.flip(img,1)
if __name__ == '__main__':
    img=cv2.imread('D:\Code\Autopilot\data\IMG\center_2021_11_01_13_15_19_637.jpg')
    imgf=flipimg(img)
    cv2.imshow('initimg',img)
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img1[:, :, 2] = 0.5 * img1[:, :, 2]
    cv2.imshow('flipimg',cv2.cvtColor(img1,cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)

    # flipimg()
