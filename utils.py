import tqdm
import torch
import scipy.io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image

def make_laplace():
    return torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def make_conv_kernel(single_layer: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((3, 3, single_layer.shape[0], single_layer.shape[1]), device = single_layer.device)
    result[0, 0] = single_layer.clone()
    result[1, 1] = single_layer.clone()
    result[2, 2] = single_layer.clone()
    return result

def pure_conv():
    from img_restore import RestoredImage
    img = plt.imread("./input/figure3.png")
    G_mat = scipy.io.loadmat('G.mat')['G']
    conv_kernel = make_conv_kernel(torch.from_numpy(G_mat)).cuda()
    img_out = RestoredImage.conv(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda(), conv_kernel)
    save_image(img_out.clamp(0, 1), "conved.png", 1)
    
def min_max_filtering(img:np.ndarray, radius: int = 2, is_max = True):
    if img.ndim < 3:
        img = img.reshape(*img.shape, 1)
    last_dim = img.shape[-1]
    img = torch.from_numpy(img)
    padded_img = F.pad(img, [0, 0, radius, radius, radius, radius], 'constant', 0)
    max_h, max_w = padded_img.shape[0] - radius, padded_img.shape[1] - radius
    output = img.clone()
    for i in tqdm.tqdm(range(radius, max_h)):
        for j in range(radius, max_w):
            center_value = padded_img[i, j, :].norm()
            rgb_values = padded_img[i - radius:i + radius + 1, j - radius:j + radius + 1, :].reshape(-1, last_dim)
            norm_values = rgb_values.norm(dim = -1)
            if is_max == True:
                norm = torch.max(norm_values)
            else:
                norm = torch.min(norm_values)
            if (norm - center_value).abs() < 3e-3:      # 就是最大值
                center_id = ((2 * radius + 1) ** 2) // 2 + 1
                _, idx = torch.kthvalue(norm_values, center_id)
                output[i - radius, j - radius, :] = rgb_values[idx, :]        # 最大值中值化
    return output

def bidirectional_filtering():
    img = plt.imread("./input/blurred3.png")
    print(img.dtype, img.shape)
    new_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    new_img[..., 0] = min_max_filtering(new_img[..., 0], 2, False)[..., 0]
    laplace = cv.Laplacian(new_img[..., 0], cv.CV_32F)
    plt.imshow(laplace)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    img_sum = torch.zeros(480, 720, 3)
    for i in range(1, 5):
        img = plt.imread(f"tensor-{i}.png")
        img_sum += img
    plt.imsave("tensor.png", img)