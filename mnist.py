from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #第一次运行会从网络下载mnist数据集并存储到~/.keras/datasets
print(train_images.shape)

from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()

#Dense层一般用来处理2D张量，第一个参数表示返回的张量的维度，relu为修正线性函数，即将负值设为0
#why 512?
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation="softmax"))
#优化器rmsprop是梯度下降的变种，损失函数categorical_crossentropy通常用于处理多酚类问题
network.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#根据损失函数的结果进行迭代
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)