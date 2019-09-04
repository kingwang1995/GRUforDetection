# -*-coding:utf-8 -*-
# @Author:king
# @time:2019/7/16 11:31
# @File:mmm.py
# @Software:PyCharm

import tensorflow as tf
import random as rd
import math
import numpy as np
import os, sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

n_step = 361
n_input = 1
n_output = 2
n_hidden = 32
n_layers = 1
n_batch = 256
# n_batch = 512
max_iter = 5000
n_display = 1
n_save = 500
lr = 1e-3
# lr = 0.0005
# data_path = "./train"
# val_path = "./val"

data_path = "./train_180116"
val_path = "./val_180116"

test_path = './test'

check_dir = "./check_GRU"
freeze_dir = check_dir + "/freeze"
logdir = './log_GRU'

gpu = "1"
test_save = './result'


class LaserData(object):
    def __init__(self, train_path, val_path):
        self.sample_names = []
        self.train_x = []
        self.train_y = []
        self.val_x = []
        self.val_y = []
        [self.train_x, self.train_y] = self.read_data(train_path)
        [self.val_x, self.val_y] = self.read_data(val_path)
        self.train_x = self.transfor_data(self.train_x)
        self.val_x = self.transfor_data(self.val_x)

    def read_data(self, filepath):
        x = []
        labels = []
        for root, sub_folder, file_list in os.walk(filepath):
            for file_path in file_list:
                sample_name = os.path.join(root, file_path)
                self.sample_names.append(sample_name)

                f = open(sample_name)
                data = f.read()
                rows = data.split('\n')
                rows = rows[0:-1]
                dist = []
                dist_label = []
                for row in rows:
                    split_row = row.split(",")
                    dist.append([float(split_row[0])])
                    cls = float(split_row[1])
                    label = []
                    if cls == 1.0:
                        label = [0.0, 1.0]
                    else:
                        label = [1.0, 0.0]
                    dist_label.append(label)

                x.append(dist)
                labels.append(dist_label)

        return [x, labels]

    def transfor_data(self, x):
        r = 3.0
        ret = (np.power(r, x) - 1.0) / (r - 1.0)
        return ret

    def next_batch(self, batch_size):
        arr = np.arange(len(self.train_x))
        np.random.shuffle(arr)
        indexs = arr[:batch_size]
        x_batch = [self.train_x[i] for i in indexs]
        y_batch = [self.train_y[i] for i in indexs]
        seq_len = [n_step for i in indexs]
        return [x_batch, y_batch, seq_len]


def get_model(inputs, labels):
    with tf.device("/gpu:0"):
        # with tf.device("/gpu:" + gpu):
        n_v = n_input
        weights = {'input': tf.Variable(tf.random_normal(shape=[n_v, n_hidden], mean=0, stddev=1.0, dtype=tf.float32)),
                   'output': tf.Variable(
                       tf.random_normal(shape=[n_hidden * 2, n_output], mean=0, stddev=1.0, dtype=tf.float32))}
        biases = {'input': tf.Variable(tf.random_normal(shape=[n_hidden])),
                  'output': tf.Variable(tf.random_normal(shape=[n_output]))}

        x = tf.transpose(inputs, [1, 0, 2])
        y = tf.transpose(labels, [1, 0, 2])
        x = tf.reshape(x, [-1, n_v])
        x = tf.matmul(x, weights['input']) + biases['input']
        x = tf.split(x, n_step)

        gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, 0.6)
        # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
        mcell = tf.contrib.rnn.MultiRNNCell(cells=[gru_cell] * n_layers, state_is_tuple=True)

        # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
        outputs, _ = tf.contrib.rnn.static_rnn(cell=mcell, inputs=x, dtype=tf.float32)
        print(outputs)
        out = tf.reshape(outputs, shape=[-1, 2 * n_hidden])
        print(out)
        pred = tf.matmul(out, weights['output']) + biases['output']
        print(pred)
        y = tf.reshape(y, [-1, n_output])
        print(y)
        correct_pred = tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), 'float')
        accuracy = tf.reduce_mean(correct_pred)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=-1)
        # cost=tf.pow(tf.reshape(pred,[n_step,-1])-tf.reshape(y,[n_step,-1]),2.0)
        cost = tf.reduce_mean(cost, 0)
        pred = tf.reshape(tf.nn.softmax(pred), shape=[n_step, -1, n_output], name='predict')
    return [pred, cost, accuracy]


x_in = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_input], name='inputs')
y_in = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_output])

loss_train_ph = tf.placeholder(tf.float32)
loss_val_ph = tf.placeholder(tf.float32)
loss_train_op = tf.summary.scalar('train_loss', loss_train_ph)
loss_val_op = tf.summary.scalar('val_loss', loss_val_ph)

data = LaserData(data_path, val_path)

[pred, loss, acc] = get_model(x_in, y_in)


def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    train_writer = tf.summary.FileWriter(logdir + '/train')
    val_writer = tf.summary.FileWriter(logdir + '/val')
    with tf.device("/gpu:0"):
        # with tf.device("/gpu:" + gpu):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        cpkt = tf.train.latest_checkpoint(check_dir)
        if cpkt:
            print('restore' + cpkt)
            saver.restore(sess, cpkt)
        tf.train.write_graph(sess.graph_def, freeze_dir, "laser_monitor-model.pbtxt", as_text=True)
        # saver.save(sess,os.path.join(freeze_dir,'wheel-model.cpkt'))

        feed_val = {x_in: data.val_x, y_in: data.val_y}
        for i in range(max_iter + 1):
            [x_batch, y_batch, _] = data.next_batch(n_batch)
            feed = {x_in: x_batch, y_in: y_batch}
            if i % n_display == 0:
                train_loss = sess.run(loss, feed_dict=feed)
                val_loss, val_acc = sess.run([loss, acc], feed_dict=feed_val)
                t_loss, v_loss = sess.run([loss_train_op, loss_val_op],
                                          feed_dict={loss_train_ph: train_loss,
                                                     loss_val_ph: val_loss})

                train_writer.add_summary(t_loss, i)
                val_writer.add_summary(v_loss, i)
                print('iter {}  train loss : {} , val loss: {} , val_accuracy: {},time_now:{}'.format(i, train_loss,
                                                                                                      val_loss,
                                                                                                      val_acc,
                                                                                                      time.strftime(
                                                                                                          "%Y-%m-%d %H:%M:%S",
                                                                                                          time.localtime())))

            if i % n_save == 0 and not (i == 0):
                if not os.path.isdir(check_dir):
                    os.mkdir(check_dir)
                print('save iter= {}'.format(i))
                saver.save(sess, os.path.join(check_dir, 'laser_monitor.cpkt'), global_step=i)

            sess.run(train_op, feed_dict=feed)
        if not os.path.isdir(freeze_dir):
            os.mkdir(freeze_dir)
        saver.save(sess, os.path.join(freeze_dir, 'laser_monitor-model.cpkt'))

        tf.summary.FileWriter(logdir, sess.graph)
    tf.reset_default_graph()


def test():
    data_test = LaserData(data_path, test_path)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    train_writer = tf.summary.FileWriter(logdir + '/train')
    val_writer = tf.summary.FileWriter(logdir + '/val')
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        cpkt = tf.train.latest_checkpoint(check_dir)
        if cpkt:
            print('restore' + cpkt)
            saver.restore(sess, cpkt)
        tf.train.write_graph(sess.graph_def, freeze_dir, "laser_monitor-model.pbtxt", as_text=True)
        feed_val = {x_in: data_test.val_x, y_in: data_test.val_y}
        prediction = sess.run(pred, feed_dict=feed_val)
        [step, n, v] = prediction.shape
        if not os.path.isdir(test_save):
            os.mkdir(test_save)

        for i in range(n):
            file = test_save + '/' + str(i) + '.txt'
            print("save result:" + file)
            src = data_test.val_x[i]
            src_label = data_test.val_y[i]
            label = prediction[:, i, :]
            with open(file, 'w') as f:
                for j in range(step):
                    line_data = str(src[j][0])
                    line_label = '0'
                    if label[j][0] < label[j][1]:
                        line_label = '1'
                    line_src_label = '0'
                    if src_label[j][0] < label[j][1]:
                        line_src_label = '1'
                    line = line_data + ',' + line_label + ',' + line_src_label + '\n'
                    f.write(line)


train()
test()