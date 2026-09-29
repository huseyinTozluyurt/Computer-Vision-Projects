import numpy as np

# Simple grayscale image (5x5)
image = np.array([
    [10, 10, 10, 10, 10],
    [10, 50, 50, 50, 10],
    [10, 50,100, 50, 10],
    [10, 50, 50, 50, 10],
    [10, 10, 10, 10, 10]
])

# Edge detection kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    
    output = np.zeros((h-2, w-2))  # no padding

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

result = convolve(image, kernel)
print(result)