#2021/10/29 20:58
import random
import numpy as np
from numpy.random import choice
import cv2.cv2 as cv2
import torch
# random.seed=13
img_shape=(128,32)
def redata_all():
    '''加载所有数据以及对应标签
    :rtype list[list]'''
    with open('../data/driving.csv','r',encoding='utf8')as f:
        data=f.read().split('\n')
    tmp0=[]
    tmp1=[]
    for i in data:
        if '.' not in i:continue
        if float(i.split(',')[-1])==0:
           tmp0.append(i)
        else:tmp1.append(i)

    tmp0=choice(tmp0,size=800,replace=False)
    # print(len(tmp1))
    # print(len(tmp0))
    data=[str(i) for i in tmp0]+tmp1
    # print(data[:10])
    random.shuffle(data)
    # print(data[:10])
    # exit()
    return [tuple(i.split(',')) for i in data]

def redata_url():
    '''加载所有数据的img路径
    :rtype list[list]'''
    with open('../data/driving.csv','r',encoding='utf8')as f:
        data=f.read().split('\n')
    return [tuple(i.split(','))[0] for i in data]

def data_trans_ls(data,url):
    '''根据选择的URL，返回list'''
    res=[]
    for d in data:
        if d[0] in url:
            res.append(d)
    return res


def load_train_val(train_weight=0.9):
    '''返回训练集数据以及验证集数据'''
    # random.seed=20
    data_all=redata_all()
    url_all=redata_url()
    data_len=len(url_all)

    train_num=int(data_len*train_weight)
    val_num=int(data_len*((1-train_weight)/2))
    test_num=data_len-train_num-val_num
    # print(train_num)
    # print(val_num)
    # print(test_num)
    train_url=choice(url_all,size=train_num,replace=False)
    train_data=data_trans_ls(data_all,train_url)
    tmp_data=list(set(url_all)-set(train_url))
    val_url=choice(tmp_data,size=val_num,replace=False)
    val_data=data_trans_ls(data_all,val_url)
    test_data=data_trans_ls(data_all,list(set(tmp_data)-set(val_url)))

    return train_data,val_data,test_data

def trans_data(imgpath):
    '''将图像转成模型输入需要的格式
    :param  图像路径
    :return Tensor（img）'''
    img = cv2.imread(imgpath)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[65:140, :]
    img = cv2.resize(img, img_shape) / 255 - 0.5
    img = np.asarray(img).transpose(-1, 1, 0)
    img = np.array([img])
    img = torch.from_numpy(img)
    img = img.float()
    return img

    # train_data=random.choices(data_all,k=int(len(data_all)*train_weight))
    # val_test_data=list(set(data_all)-set(train_data))
    # val_data=random.choices(val_test_data, k=int(len(val_test_data)*((1-train_weight)/2)) )

    # return train_data,val_data,list(set(val_test_data)-set(val_data))
if __name__ == '__main__':
    train,val,test=load_train_val()
    print(len(train))
    print(train[:10])
    print(len(val))
    print(val[:10])
    print(len(test))
    print(test[:10])