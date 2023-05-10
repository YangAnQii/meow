import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
from swdpytorchmaster import swd
from fixnoise import fn

# Set random seed for reproducibility
manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)#和torch.rand()、torch.randn()等函数搭配使用。通过指定seed值，可以令每次生成的随机数相同。
##定义部分参数
# Root directory for dataset
dataroot = "E:\pythonProject\LIDC-IDRI-train"
# Batch size during training
batch_size = 2
#batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
#image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
#ndf = 64
# Number of training epochs
num_epochs = 75
#num_epochs = 15
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               #transforms.Grayscale(num_output_channels=3),
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
'''
len(dataloader)=136:一共272张图片，batch_size=2，一共136个batches
'''
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()#显示数据集图片

def weights_init(m):##权重初始化函数
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:#classname中有”Conv"
        nn.init.normal_(m.weight.data, 0.0, 0.02)##m.weight.data:torch.Tensor
    elif classname.find('BatchNorm') != -1:#classname中有”BatchNorm"
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

##生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator().to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)
# Print the model
# print(netG)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), ##ndf是filter的数量，因为算的时候，就是先对每一个滤波器去计算每一个通道的，然后把每个通道的结果加和，这个和就是这个滤波器的结果，输出的就是由ndf个滤波器算出的结果的向量。
            nn.LeakyReLU(0.2, inplace=True),##32,32,14
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),##64,64,7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)



def to_img(x):

    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)##将范围限定在[0,1]之间
    x = x.view(x.shape[0], 3, 64, 64)
    return x

# Create the Discriminator
netD = Discriminator().to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
# print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = fn ##固定不变
print(fixed_noise)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
swd_values = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        '''
        data:list,
        '''

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()  ##每轮epoch开始把D模型的参数梯度设成0(如果不清零，那么使用的这个grad就与上一个mini-batch有关)
        # Format batch
        real_cpu = data[0].to(device)##将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        #print("data=",data)
        #data[0]=img,data[1]=label(无label，全为0)
        #print("real_cpu=",real_cpu)
        # print("real_cpu.size=",real_cpu.size())

        '''
        real_cpu.size= torch.Size([128, 3, 64, 64])
        '''

        b_size = real_cpu.size(0) ##返回real_cpu第一维的维度数
        #print("b_size=",b_size)

        '''
        b_size=128=batch_size
        '''

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        ##生成一个b_size(128)大小、由real_label填充的tensor(label)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)##将输出转化为1维并自动排序
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)##计算交叉熵
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()##对output求均值后返回(.item()用于显示更高精度)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)#生成维度b_size*nz的noise
        # Generate fake image batch with G
        fake = netG(noise)##放入生成器
        label.fill_(fake_label)##用fake_label填充label张量
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)##detach()使fake不能求导，即停止生成器求导，先训练判别器
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()##此时生成器不能求导，只在训练判别器
        D_G_z1 = output.mean().item()##对output求均值后返回(.item()用于显示更高精度)
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake##判别器的损失包含真判断为假、假判断为真的损失总和

        '''
        pyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
        具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算。
        (.detach()函数就是将requires_grads改为FALSE)
        如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
        因此，如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。
        '''

        # Update D
        optimizerD.step()##根据梯度更新D（判别器）参数

        '''
        step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。
        因为梯度下降是基于梯度的，所以 在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
        '''

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        '''
        引入swd作为损失函数的一部分值进行训练，如下
        '''
        lum = 0.0005

        def extract64img(dirPath):
            """
            :param dirPath: 文件夹路径
            :return:
            """
            # 对目录下的文件进行遍历
            tensor_list = list()
            pathDir = os.listdir(dirPath)
            imgnum = len(pathDir)

            if imgnum > 64:  ##如果是训练集，则只取其中64张图片
                sample = random.sample(pathDir, 64)
                for j in sample:
                    # 判断是否是文件
                    if os.path.isfile(os.path.join(dirPath, j)) == True:
                        a = os.path.basename(j)
                        name = dirPath + '\\' + a
                        img = cv.imread(name)  ##此时img.shape为(64,64,3)
                        transf = transforms.ToTensor()
                        img_tensor = transf(img)  ##img_tensor.size()=torch.Size([3, 64, 64])
                        tensor_list.append(img_tensor)

            elif imgnum == 64:  ##fake图片共有64张
                for j in pathDir:
                    # 判断是否是文件
                    if os.path.isfile(os.path.join(dirPath, j)) == True:
                        a = os.path.basename(j)
                        name = dirPath + '\\' + a
                        img = cv.imread(name)
                        transf = transforms.ToTensor()
                        img_tensor = transf(img)
                        # img=cv2.resize(img,(100,100))#使尺寸大小一样
                        tensor_list.append(img_tensor)

            final_list = torch.stack(tensor_list)
            # print(final_list.size()),torch.Size([64, 3, 64, 64])
            return final_list

        path1 = 'E:\pythonProject\LIDC-IDRI-train\\5'
        img1list = extract64img(path1)  ##img1list.shape=(64,64,64,3),272个训练数据中随机提取64个

        out = swd.swd(img1list, netG(fixed_noise), n_pyramids=1, device="cpu")*lum
        print(out.item())

        '''
        swd引入部分结束
        '''
        netG.zero_grad()##每轮epoch开始把G模型的参数梯度设成0(如果不清零，那么使用的这个grad就与上一个mini-batch有关)
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)##将生成的假图片放入D获得判别值（是否骗过了判别器）
        # Calculate G's loss based on this output
        errG = criterion(output, label)##损失为D给出的判别值与real_label的交叉熵
        '''
        增添了swd作为损失函数
        '''
        # Calculate gradients for G
        errG.backward()##对损失函数可求导参数求导
        D_G_z2 = output.mean().item()##对output求平均（.item()用于显示更高精度）
        # Update G
        optimizerG.step()##更新各个参数

        # Output training stats
        if i % 50 == 0:
        #每训练50个batches输出一次结果
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())##给G_losses后面追加一个errG.item(),用于显示更高精度
        D_losses.append(errD.item())
        swd_values.append(out.item())

        ##每隔20次输出一次结果
        if (epoch*136+i+1)%20==0:
         print('img=',epoch*136+i+1)
         save = to_img(netG(fixed_noise).cpu().data)

         #print(netG(fixed_noise).size()) torch.Size([64, 3, 64, 64])

         # if epoch*136+i+1>=1000:
         #     if not os.path.exists('E:/pythonProject/test/lidc-idri/'+str(epoch*136+i+1)):
         #         os.mkdir('E:/pythonProject/test/lidc-idri/'+str(epoch*136+i+1))
         #     l=0
         #     while l<64:
         #         save_image(save[l], 'E:/pythonProject/test/lidc-idri/'+str(epoch*136+i+1)+'/image_{}.png'.format(l+1))
         #         l=l+1

         # print(save.size()) ##torch.Size([64, 3, 64, 64]),64张[3,64,64]的生成图
         # plt.imshow(np.transpose(save[0],(1,2,0)))
         # plt.show()
         if not os.path.exists('E:\pythonProject\\result\\5\lidc-idri-without-swd'):
             os.mkdir('E:\pythonProject\\result\\5\lidc-idri-without-swd')
         save_image(save, 'E:/pythonProject/result/5/lidc-idri-without-swd/image_{}.png'.format(int((epoch * 136+ i + 1)/20)))#保存每次生成的图片
         # if not os.path.exists('./dcgan_flower2_img'):
         #     os.mkdir('./dcgan_flower2_img')
         # save_image(save, './dcgan_flower2_img/image_{}.png'.format(epoch*256+i + 1))

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
        ##每训练100次，或已经训练完毕时
            with torch.no_grad():##表明当前计算不需要反向传播
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            ##make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽。
        if ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):##保留训练结束后的最后一组生成图
            if not os.path.exists('E:/pythonProject/result/5/lidc-idri-without-swd/fake'):
                os.mkdir('E:/pythonProject/result/5/lidc-idri-without-swd/fake')
            l=0
            while l<64:
                save_image(save[l], 'E:/pythonProject/result/5/lidc-idri-without-swd/fake/image_{}.png'.format(l+1))
                l=l+1

        iters += 1


plt.subplot(2, 1, 1)
# plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")

plt.subplot(2, 1, 2)
#plt.figure(figsize=(10,5))
plt.title("Changes in SWD During Training")
plt.plot(swd_values,label="SWD")
plt.xlabel("iterations")
plt.ylabel("Value")

plt.legend()
plt.tight_layout(pad=1.5)
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
#np.transpose(i,(1,2,0))调换i的维度,animated动画化
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save('LIDC-IDRI-without-swd.gif')