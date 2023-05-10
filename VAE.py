import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import matplotlib.animation as animation
from fixnoise import vaefn
import random

class MyDataset(Dataset):
    def __init__(self,path_dir,transform=None):
        self.path_dir=path_dir
        self.transform=transform
        self.image=os.listdir(self.path_dir)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_index=self.image[index]
        img_path=os.path.join(self.path_dir,image_index)
        img = Image.open(img_path).convert('RGB')
        label = "5"
        if self.transform is not None:
            img=self.transform(img)
        return img,label

path_dir='E:\pythonProject\LIDC-IDRI-train\\5'
image=os.listdir(path_dir)
im_tfs= tfs.Compose([tfs.ToTensor(),
                             tfs.Resize(64),
                             tfs.CenterCrop(64),
                             tfs.Grayscale(3),
                             tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 标准化,取值范围变为[-1,1]
                            ])
train_set=MyDataset('E:\pythonProject\LIDC-IDRI-train\\5',transform=im_tfs)
img=train_set[0]
train_data=DataLoader(train_set,batch_size=20,shuffle=True)
imgs = train_set

_, axes=plt.subplots(3, 4)##sbuplots'return:fig, axs
for i in range(3):##range(i)指从0至i-1,不包含i
    for j in range(4):
        axes[i][j].imshow(imgs[i*3 + j][0].permute(1,2,0))
        print(np.shape(imgs[i*3 + j][0]),np.shape(imgs[i*3 + j][0].permute(1,2,0)))
        axes[i][j].get_xaxis().set_visible(False)
        axes[i][j].get_yaxis().set_visible(False)
plt.show()

##定义VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(4096, 784)
        self.fc2 = nn.Linear(784, 400)
        self.fc31 = nn.Linear(400, 20) # mean
        self.fc32 = nn.Linear(400, 20) # var
        self.fc4 = nn.Linear(20, 400)
        self.fc5 = nn.Linear(400, 784)
        self.fc6 = nn.Linear(784,4096)

    def forward(self, x):
        mu, logvar = self.encode(x) # 编码
        # note = self.determine(logvar)
        # if note == 1:
        #   print("note = ",note)
        # std = logvar.mul(0.5).exp_()  # e**(x*0.5)
        # print("std = ", std.size(), "mu = ", mu.shape)
        #   eps = torch.FloatTensor(std.size()).normal_()
        #   fixz = eps.mul(std).add_(mu)
        #   return self.decode(fixz)  # 解码，同时输出均值方差
        # else:
        #   print("note = ", note)
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z), mu, logvar  # 解码，同时输出均值方差


    def encode(self, x): #编码层
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def decode(self, z):#解码层
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return F.tanh(self.fc6(h4))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() #e**(x*0.5)
        eps = torch.FloatTensor(std.size()).normal_()
        #print("std.size()=",std.size())
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        # print("std = ",std.size(),"mu = ",mu.shape)
        return eps.mul(std).add_(mu)

    # def determine(self,logvar):
    #     std = logvar.mul(0.5).exp_()  # e**(x*0.5)
    #     print("std.size=",std.size()[0])
    #     if std.size()[0]==64:
    #         note = 1
    #     else:
    #         note = 0
    #     return note



net = VAE() # 实例化网络
img_list = []
lossvalue = []
if torch.cuda.is_available():
    net = net.cuda()


x, _ = train_set[2] # (3,60,60)
x = x.view(x.shape[0], -1)#(3,3600)
if torch.cuda.is_available():
    x = x.cuda()
x = Variable(x)
_, mu, var = net(x)##通过Model的call函数调用了forward
# print(_.size())


'''
   定义一个函数将最后的结果转换回图片
'''
def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 3, 64, 64)
    x = tfs.Grayscale(3)(x)
    #print("shape:",x.shape," 图片数量：",x.shape[0])
    return x

reconstruction_function = nn.MSELoss(size_average=False)


'''
   定义损失函数
'''
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)




for e in range(4000):
    for im, _ in train_data:##im图片，_标签
        im = im.view(im.shape[0],3,-1)
        im = Variable(im)
        if torch.cuda.is_available():
            im = im.cuda()
        recon_im, mu, logvar = net(im)##recon_im为编码再解码后的im
        # img_list.append(vutils.make_grid(recon_im, padding=2, normalize=True))
        loss = loss_function(recon_im, im, mu, logvar) / im.shape[0] # 将 loss 平均
        print("loss = ",loss.item(),"e=",e," (e + 1) % 20= ",(e + 1) % 20)
        if e>=100:
            lossvalue.append(loss.item())
        optimizer.zero_grad()##把模型的参数梯度设成0
        loss.backward()##计算当前梯度，反向传播
        optimizer.step()##根据梯度更新参数

    if (e+ 1) % 20 == 0:##每隔20个epoch输出一次结果
        '''
        '''
        sample = random.sample(image, 64)
        s = []
        for j in sample:
            # 判断是否是文件
            if os.path.isfile(os.path.join(path_dir, j)) == True:
                a = os.path.basename(j)
                name = path_dir + '\\' + a
                img = cv.imread(name)  ##此时img.shape为(64,64,3)
                transf = transforms.ToTensor()
                img_tensor = transf(img)  ##img_tensor.size()=torch.Size([3, 64, 64])
                s.append(img_tensor)
        s = torch.stack(s)
        # print("s.shape=",s.shape),torch.Size([64, 3, 64, 64])
        s = s.view(s.shape[0], 3, -1)
        # print("s.shape=", s.shape),torch.Size([64, 3, 4096])
        s = Variable(s)
        if torch.cuda.is_available():
            s = s.cuda()
        fake,_,_ = net(s)  ##recon_im为编码再解码后的im
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        '''
        用noise生成64张图片
        '''
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.item()))
        save = to_img(fake.cpu().data)
        if not os.path.exists('E:\pythonProject\\result\\5\VAE'):
            os.mkdir('E:\pythonProject\\result\\5\VAE')
        save_image(save, 'E:\pythonProject\\result\\5\VAE\image_{}.png'.format(int((e + 1)/20)))
    if (e == 3999):  ##保留训练结束后的最后一组生成图
        if not os.path.exists('E:/pythonProject/result/5/VAE/fake'):
            os.mkdir('E:/pythonProject/result/5/VAE/fake')
        l = 0
        while l < 64:
            save_image(save[l], 'E:/pythonProject/result/5/VAE/fake/image_{}.png'.format(l + 1))
            l = l + 1

plt.figure(figsize=(10,5))
plt.title("Loss During Training")
plt.plot(lossvalue)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.tight_layout(pad=1.5)
plt.show()


fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
#np.transpose(i,(1,2,0))调换i的维度,animated动画化
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save('LIDC-IDRI.gif')