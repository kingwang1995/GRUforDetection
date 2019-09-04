# -*-coding:utf-8 -*-
# @Author:king
# @time:2019/7/16 10:54
# @File:test.py
# @Software:PyCharm

import tensorflow as tf
import numpy as np

# x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
# x = np.array(x)
# print(x.shape)
# # x = tf.reshape(x, [-1, 2])
# # print(x)
# y = tf.transpose(x, [1, 0])
# y = tf.reshape(y, [-1, 2])
# print(y)

# from tensorflow.examples.tutorials.mnist import input_data
#
# # mnist是一个轻量级的类，它以numpy数组的形式存储着训练，校验，测试数据集  one_hot表示输出二值化后的10维
# mnist_data_folder = "/data"
# mnist = input_data.read_data_sets(mnist_data_folder, one_hot=True)
# print(mnist.train.labels.shape)
# x_batch,y_batch = mnist.train.next_batch(batch_size=256)
# print(x_batch.shape)
# print(y_batch.shape)
import tensorflow as tf
import numpy as np

# A = np.array([[1, 2, 3], [4, 5, 6]])
# # X = tf.transpose(A, [0, 1])
# # with tf.Session() as sess:
# #     print("original:", A)
# #     print("tranpose:", sess.run(X))
# kfff = tf.cast(True, 'float')
# ssss = tf.cast(False, 'float')
# with tf.Session() as sess:
#     print(sess.run(kfff))
#     print(sess.run(ssss))
b = tf.constant([[1, 2, 3], [3, 2, 1], [4, 5, 6], [6, 5, 4]])
a = tf.random_normal(shape=[32 * 2, 2], mean=0, stddev=1.0, dtype=tf.float32)
with tf.Session() as sess:
    # print(sess.run(b))
    # print(sess.run(tf.argmax(b, 0)))
    # print(sess.run(tf.argmax(b, 1)))
    print(sess.run(a))
