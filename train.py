#2021/10/29 19:16
from cv2 import cv2
import torch
import random
import torch.nn as nn
from torch import optim as opt
from torch.utils.tensorboard import SummaryWriter
from models.custom_model import CustomerNet
from src.Load_data import load_train_val,trans_data
import numpy as np
from tqdm.std import tqdm
img_shape=(128,32)

def save(model,fi):
    torch.save(model.state_dict(),f'../weight/runs/train_{fi}.pt')
    print('Finished Training')

def main():
    model=CustomerNet()
    #加载损失函数
    # loss=nn.CrossEntropyLoss()
    loss=nn.MSELoss()
    #加载优化器
    optmizer=opt.SGD(model.parameters(),lr=0.003,momentum=0.8)
    # trans=transforms.Compose([transforms.ToTensor()])
    train_data,val_data,test_data=load_train_val(1)
    sumwriter=SummaryWriter(log_dir='../logs')
    #训练论数
    epochs=80
    step=1
    for epoch in range(epochs):
        print(f'-------------第{epoch+1}轮训练-----------')
        runing_loss = 0
        ind=0
        train_data, val_data, test_data = load_train_val(0.8)
        random.shuffle(train_data)
        for data in tqdm(train_data):
            try:imgpath,lables=data
            except:continue
            lables=torch.from_numpy(np.array([float(lables)],dtype=np.float32))
            #图像预处理
            img=trans_data(imgpath)
            optmizer.zero_grad()
            netout=model(img)
            # print(lables)
            lossnow=loss(netout,lables)
            lossnow.backward()
            optmizer.step()
            runing_loss+=lossnow.item()
            ind += 1
            if ind%1000==0:
                print('<<<<--------第%3d轮,图片：%10d张---------->>>> loss:%3f' %(epoch+1,ind,runing_loss/1000))
                # sumwriter.add_image('img',img,dataformats='CWH')
                sumwriter.add_scalar(tag='Loss',scalar_value=runing_loss/1000,global_step=step)
                step+=1
                if runing_loss/1000==0:
                    save(model,epoch)
                    return
                runing_loss = 0
        save(model,epoch)




if __name__ == '__main__':
    main()