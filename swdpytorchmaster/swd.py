from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0  ##高斯核
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k

def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)        
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]##分通道卷积（下采样）
    down_image = torch.cat(multiband, dim=1)##cat:在给定维度上对输入的张量序列seq进行连接操作(把分别下卷积的部分连接起来)
    return down_image

def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)  ##实现插值和上采样，scale_factor：指定输出为输入的多少倍数
    multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)] ##对各个通道分别卷积

    '''
    F.conv2d:
    input:input tensor of shape (minibatch,in_channels,iH,iW)
    weight:filters of shape(out_channels,in_channels/groups,kH,kW)
    stride:the stride of the convolving kernel. Can be a single number or a tuple (sH, sW).Default: 1
    padding:implicit paddings on both sides of the input.
    '''
    up_image = torch.cat(multiband, dim=1)  ##在给定维度上对输入的张量序列seq进行连接操作,输出tensor
    return up_image

def gaussian_pyramid(original, n_pyramids, device="cpu"):##高斯图像金字塔
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):##下采样次数
        x = pyramid_down(x, device=device)
        pyramids.append(x)  ##下采样一次，记录一次数据
    return pyramids  ##一个batch的所有图片下采样结果

def laplacian_pyramid(original, n_pyramids, device="cpu"): ##original:每一batch的所有图片
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])        
    return laplacian

def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size) ##一个batch一组，总组数
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size] ##取出该组的所有image
        p = laplacian_pyramid(x.to(device), n_pyramids, device=device) ##得到该batch的拉普拉斯金字塔
        p = [x.cpu() for x in p]
        pyramids.append(p) ##所有batch的拉普拉斯金字塔
    del x
    result = []
    for i in range(n_pyramids + 1):##金字塔层数
        x = []
        for j in range(n):##组数
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result ##所有图片的拉普拉斯金字塔

def extract_patches(pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
    assert pyramid_layer.ndim == 4
    n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
    # random slice 7x7
    p_slice = []
    for i in range(n):
        # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
        ind_start = i * unfold_batch_size ##开始图片编号
        ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0)) ##结束图片编号
        x = pyramid_layer[ind_start:ind_end].unfold(
                2, slice_size, 1).unfold(3, slice_size, 1).reshape(
                ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size) ##将tensor.shape转化为(batch num,
                                                                                        ##chanel,-1,slice-size,slice-size)
        '''
        关于unfold:手动实现的滑动窗口操作,只卷不积
        ret = x.unfold(dim, size, step) 
        dim：int，表示需要展开的维度(可以理解为窗口的方向)
        size：int，表示滑动窗口大小
        step：int，表示滑动窗口的步长
        '''
        # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
        x = x[:,:, slice_indices,:,:]
        # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
        p_slice.append(x.permute([0, 2, 1, 3, 4]))
    # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
    x = torch.cat(p_slice, dim=0)##合并batch
    # normalize along ch
    std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True) ##返回 input 张量中所有元素的标准差和均值
    x = (x - mean) / (std + 1e-8) ##归一化x(所有图片)
    # reshape to 2rank
    x = x.reshape(-1, 3 * slice_size * slice_size)
    return x
        
def swd(image1, image2, 
        n_pyramids=None, slice_size=7, n_descriptors=128,
        n_repeat_projection=128, proj_per_repeat=4, device="cpu", return_by_resolution=False,
        pyramid_batchsize=128):
    # n_repeat_projectton * proj_per_repeat = 512
    # Please change these values according to memory usage.
    # original = n_repeat_projection=4, proj_per_repeat=128    
    assert image1.size() == image2.size()
    assert image1.ndim == 4 and image2.ndim == 4

    if n_pyramids is None:
        n_pyramids = int(np.rint(np.log2(image1.size(2) // 16)))
    with torch.no_grad():
        # minibatch laplacian pyramid for cuda memory reasons
        pyramid1 = minibatch_laplacian_pyramid(image1, n_pyramids, pyramid_batchsize, device=device)
        pyramid2 = minibatch_laplacian_pyramid(image2, n_pyramids, pyramid_batchsize, device=device)
        ##分别得到两组图片的拉普拉斯金字塔
        result = []

        for i_pyramid in range(n_pyramids + 1):##每层拉普拉斯金字塔
            # indices
            n = (pyramid1[i_pyramid].size(2) - 6) * (pyramid1[i_pyramid].size(3) - 6)##该层金字塔的h-6,w-6相乘
            indices = torch.randperm(n)[:n_descriptors] ##torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列
                                                        ##[:n_descriptors]：取前n_descriptors个
            # extract patches on CPU
            # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
            p1 = extract_patches(pyramid1[i_pyramid], indices,
                            slice_size=slice_size, device="cpu")
            p2 = extract_patches(pyramid2[i_pyramid], indices,
                            slice_size=slice_size, device="cpu")##第i层拉普拉斯金字塔，抽取序列为indices,归一化，返回(-1, 3 * slice_size * slice_size)

            p1, p2 = p1.to(device), p2.to(device)

            distances = []
            for j in range(n_repeat_projection):##随机投影的次数
                # random
                rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
                rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
                # projection
                proj1 = torch.matmul(p1, rand)
                proj2 = torch.matmul(p2, rand)
                proj1, _ = torch.sort(proj1, dim=0)
                proj2, _ = torch.sort(proj2, dim=0)
                d = torch.abs(proj1 - proj2)
                distances.append(torch.mean(d))

            # swd
            result.append(torch.mean(torch.stack(distances)))
            '''
            stack:把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
            '''
        
        # average over resolution
        result = torch.stack(result) * 1e3
        if return_by_resolution:
            return result.cpu()
        else:
            return torch.mean(result).cpu()
