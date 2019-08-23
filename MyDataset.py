from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# with open(r'C:\Users\uestc\Desktop\项目\DatasetUtil\dataset\label.text') as f:
#     for line in f:
#         #固定用法
#         line = line.strip('\n')
#         line = line.rstrip()
#         words=line.split()


def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self,label_path,transform=None,target_transform=None,loader=default_loader):

        self.images=[]
        self.labels=[]
        self.filenames=[]

        with open(label_path) as f:
            for line in f:
                #固定用法
                line = line.strip('\n')
                line = line.rstrip()
                words=line.split()
                self.images.append(words[0])
                self.labels.append(words[1])
                self.filenames.append(words[0])

        self.transform=transform
        self.target_trandform=target_transform
        self.loader=loader

    def __getitem__(self, index):
        fn = self.images[index]
        label=int(self.labels[index])
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label,self.filenames[index]

    def __len__(self):
        return len(self.images)


# def test():
#     test_dataset=MyDataset(label_path=r'C:\Users\uestc\Desktop\项目\DatasetUtil\dataset\label.text',transform=transforms.ToTensor())
#     test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=2)
#     for _,(b_x,b_y,b_name) in enumerate(test_loader):
#
# if __name__=='__main__':
#     test()