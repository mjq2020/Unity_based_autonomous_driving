#2021/10/30 12:03
import torch
from cv2 import cv2
from tqdm.std import tqdm
from models.custom_model import CustomerNet
from src.Load_data import load_train_val

img_shape=(128,32)
def load_test_data():
    train,val,test=load_train_val()
    return test

def trans_data(imgpath):
    img=cv2.imread(imgpath)
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img[65:140,:]

    img=cv2.resize(img,img_shape)/255-0.5


    img = torch.from_numpy(img)
    img=img.float()
    img=torch.reshape(img,(1,3,128,32))


    return img

def detecter():
    print('开始加载模型！')
    model=CustomerNet()
    model.load_state_dict(torch.load('../weight/runs/train_79.pt'))
    print(model)
    print('模型加载完成！')
    print('开始加载数据！')
    test_data=load_test_data()
    print('数据加载完成！')
    for i in test_data:
        img,lable=i
        lable=float(lable)
        img=trans_data(img)
        output=model(img)

        print('真实值：%6f，预测值：%6f，相差：%6f' %(float(lable),output,lable-float(output)))
        # print(output,lable)



if __name__ == '__main__':
    detecter()