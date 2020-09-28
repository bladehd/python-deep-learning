from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #第一次运行会从网络下载mnist数据集并存储到~/.keras/datasets
print(train_images.ndim)#轴的个数
print(train_images.shape)#形状
print(train_images.dtype)#数据类型
#train_images是一个3D张量，亦是一个60000个矩阵组成的数组，每个矩阵由28 * 28个整数组成(一张灰度图像)
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()