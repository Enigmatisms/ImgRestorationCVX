import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# img is of shape (480, 720)
def convolve(img: np.ndarray, kernel: np.ndarray):
    ksize = kernel.shape[0] >> 1
    img = cv.copyMakeBorder(img, ksize, ksize, ksize, ksize, cv.BORDER_CONSTANT)

    X = cp.Variable((480, 720, 3))
    print(X)


if __name__ == "__main__":
    img = plt.imread("blurred1.png")
    convolve(img, np.random.rand(7, 7))
