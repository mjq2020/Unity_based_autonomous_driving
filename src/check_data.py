#2021/10/29 16:05
# from matplotlib.pyplot import hist
from matplotlib import pyplot as plt
from src.flip_img import filpimgs,flipimg
from cv2 import cv2
import csv
from tqdm.std import tqdm

class Check_data():
    def __init__(self):
        street=self.readcsv(3)
        print(street)
        # flipimg(street)
        # self.show_dist(street.values())
        # self.stre_num(street.values())
        # filpimgs(street)

        # self.gener_data(street)

    def gener_data(self,data:dict):
        '''产生新数据，数据量不够，自定义合理数据'''
        for i in tqdm(data.keys()):
            img=cv2.imread(i)
            if data[i]==0:
                self.write_data([i,data[i]])
            else:
                img=flipimg(img)
                imgfile='flip_'+i.split('\\')[-1]
                writepath='D:\Code\Autopilot\data\IMG\\'+imgfile
                cv2.imwrite(f'../data/IMG/{imgfile}',img)
                self.write_data([writepath,-data[i]])
                self.write_data([i,data[i]])

    def write_data(self,data:list):
        '''将新生成的数据写入csv文件'''
        with open('../data/driving.csv','a',encoding='utf-8',newline='')as f:
            ff=csv.writer(f)
            ff.writerow(data)

    def readcsv(self,num):
        '''读取原始csv文件内容，提取对应数据
        :rtype dict'''
        with open('../data/driving_log.csv','r',encoding='utf8')as f:
            data=f.read().split('\n')
        res={}
        for i in data:
            tmp=i.split(',')
            try:res[tmp[0]]=float(tmp[num])
            except:continue
        return res

    def show_dist(self,data):
        '''画出直方图'''
        plt.hist(data,log=False,bins=30,align='mid',density=True)
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('./test.jpg')
        plt.show()

    def stre_num(self,data):
        '''统计图像数据，查看左右转向数量'''
        tottle=[]
        up=[]
        for i in data:
            if  i<0:
                tottle.append(i)
            elif 0==i:continue
            else:
                up.append(i)
        print(up)
        print(len(up))
        print(tottle)
        print(len(tottle))



if __name__=='__main__':
    Check_data()