# coding:utf-8
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import train
from torch.autograd import Variable
from torchvision.models import resnet50
# �ָ�ģ��ѵ��
from 二分类尝试.train import restore
import MyDataset
BATCH_SIZE=16
default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

test_dataset=MyDataset.MyDataset(label_path=r'C:\Users\uestc\Desktop\项目\二分类尝试\dataset\label.text',transform=default_transform)
test_loader=Data.DataLoader(test_dataset,batch_size=8,shuffle=True,num_workers=2)
def start():
    resnet_model = resnet50(num_classes=2).cuda()
    restore(resnet_model)
    with torch.no_grad():
        number=0
        for step, (b_x1, b_y1,b_name) in enumerate(test_loader):
            b_x1 = Variable(b_x1.cuda())
            resnet_model.eval()
            preValue = resnet_model(b_x1)
            predict = torch.max(preValue, 1)[1].cpu().data.numpy()
            for i in range(len(b_x1)):
                if predict[i]!=b_y1[i].data.numpy():
                    print(b_name[i])
                    number+=1
            # print("predit:\n",predict,"\nactually:\n",b_y1.data.numpy())
    print(number)
def application():
    start()

def main():
    application()


if __name__ == '__main__':
    main()