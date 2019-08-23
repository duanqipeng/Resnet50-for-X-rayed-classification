import torch
import os
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torch.autograd import Variable
import random
from PIL import Image

EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
LR = 0.01  # learning rate

default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(r'C:\Users\uestc\Desktop\项目\二分类尝试\train_dataset', default_transform)
train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, )

test_dataset = torchvision.datasets.ImageFolder(r'C:\Users\uestc\Desktop\项目\二分类尝试\test_dataset', default_transform)
test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, )


def save(net):
    torch.save(net.state_dict(), 'net_params.pkl')  # save only the parameters
    print('save-success')


def restore(net):
    if (os.path.exists('net_params.pkl')):
        net.load_state_dict(torch.load('net_params.pkl'))
        print('success restore')
    else:
        print('No pkl,create')


def train():
    resnet_model = resnet50(num_classes=2).cuda()
    restore(resnet_model)

    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    # training and testing
    for epoch in range(EPOCH):
        resnet_model.train()
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = Variable(b_x.cuda())
            b_y = Variable(b_y.cuda())
            # print(b_y.shape)
            output = resnet_model(b_x)  # cnn output
            # print()
            # print(output.cpu().data.numpy())
            # print(b_y.cpu())
            # print()
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # with torch.no_grad():
            #     # testPic=pre_pic(r'C:\Users\uestc\Desktop\项目\二分类尝试\train_dataset\joint\pic330.jpg')
            #     for step, (b_x1, b_y1) in enumerate(test_loader):
            #         b_x1 = Variable(b_x1.cuda())
            #         resnet_model.eval()
            #         preValue = resnet_model(b_x1)
            #         print(b_y1)
            #         print("The prediction number is:", preValue.cpu().data.numpy())
            if step % 50 == 0:
                # for _, (t_x, t_y) in enumerate(test_loader):
                # for step, (t_x, t_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
                # t_x = Variable(t_x.cuda())
                # t_y = Variable(t_y.cuda())
                # test_output = resnet_model(t_x)
                # pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                # accuracy = float((pred_y == t_y.cpu().data.numpy()).astype(int).sum()) / float(t_y.size(0))
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())
                save(resnet_model)
    save(resnet_model)


if __name__ == '__main__':
    train()
