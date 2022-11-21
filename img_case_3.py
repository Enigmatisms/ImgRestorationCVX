import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2 as cv

def min_max_filtering(img:np.ndarray, radius: int = 2, is_max = True):
    if img.ndim < 3:
        img = img.reshape(*img.shape, 1)
    last_dim = img.shape[-1]
    img = torch.from_numpy(img)
    padded_img = F.pad(img, [0, 0, radius, radius, radius, radius], 'constant', 0)
    print(padded_img.shape)
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

def h_layer_process(img: np.ndarray):
    print(img.dtype, img.shape)
    h_layer = min_max_filtering(img[..., 0], 2, False)[..., 0]
    laplace = cv.Laplacian(h_layer, cv.CV_32F)
    plt.imshow(laplace)
    plt.colorbar()
    plt.show()

def s_layer_process(img: np.ndarray):
    s_layer = img[..., 1]
    # smooth_s_layer = cv.bilateralFilter(s_layer, 5, 30, 5)
    # s_layer = cv.medianBlur(s_layer, 5)
    # smooth_s_layer = cv.bilateralFilter(s_layer, 5, 50, 7)
    plt.figure(0)
    plt.imshow(s_layer)
    # plt.colorbar()

    # plt.figure(1)
    # plt.imshow(smooth_s_layer)
    # plt.colorbar()

    laplace = cv.Laplacian(s_layer, cv.CV_32F)
    laplace_smooth =cv.bilateralFilter(laplace, 5, 50, 7)
    plt.figure(2)
    plt.imshow(laplace_smooth)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    img = plt.imread("./input/blurred3.png")
    new_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    s_layer_process(new_img)
    