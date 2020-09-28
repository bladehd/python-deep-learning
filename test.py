import numpy as np
import tensorflow as tf


# h = tf.constant([123,456],dtype = tf.int32)
# f = tf.cast(h,tf.float32)
# print(h.dtype, f.dtype)

y = tf.constant([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])
print(y.shape)