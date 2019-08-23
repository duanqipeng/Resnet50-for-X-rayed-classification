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
BATCH_SIZE=16
default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
test_dataset = torchvision.datasets.ImageFolder(r'C:\Users\uestc\Desktop\项目\二分类尝试\hardtest', default_transform)
test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, )
def start():
    resnet_model = resnet50(num_classes=2).cuda()
    restore(resnet_model)
    with torch.no_grad():
        # testPic=pre_pic(r'C:\Users\uestc\Desktop\项目\二分类尝试\train_dataset\joint\pic330.jpg')
        for step, (b_x1, b_y1) in enumerate(test_loader):
            b_x1 = Variable(b_x1.cuda())
            resnet_model.eval()
            preValue = resnet_model(b_x1)
            predict = torch.max(preValue, 1)[1].cpu().data.numpy()
            print("predit:\n",predict,"\nactually:\n",b_y1.data.numpy())
            # print(type(b_y1[0]))
            # accuracy = float((predict == b_y1.data.numpy()).astype(int).sum()) / float(b_y1.size(0))
            # print('| test accuracy: %.2f' % accuracy)
# def pre_pic(picName):
#     img = Image.open(picName)
#     loader = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor()])
#     image = loader(img).unsqueeze(0)
#     return image.to(torch.float)


def application():
    start()

def main():
    application()


if __name__ == '__main__':
    main()
