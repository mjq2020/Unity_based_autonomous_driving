#2021/10/29 15:33
from cv2 import cv2

def readcsv(filepath):
    '''读取csv文件，
    :return 每行的数据
    :rtype list'''
    # with open(filepath,'r',encoding='utf-8')as f:
    #     data=f.read().replace('E:\data\AutoJS\\', 'D:\Code\Autopilot\data\\')
    with open(filepath,'r',encoding='utf-8')as f:
        # f.write(data)
        data=f.read().split('\n')
    return data

def image_show(filels):
    for file in filels:
        img=cv2.imread(file)
        cv2.imshow('img',img[60:140,:])
        cv2.waitKey(30)


if __name__ == '__main__':
    filels=readcsv('../data/driving_log2.csv')
    print(filels[0].split(','))
    image_show([i.split(',')[0] for i in filels])