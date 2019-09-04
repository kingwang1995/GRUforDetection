# -*-coding:utf-8 -*-
# @Author:king
# @time:2019/7/16 13:40
# @File:gru.py
# @Software:PyCharm

import tensorflow as tf
import numpy as np
import os

n_step = 361
n_input = 1
n_output = 2
n_hidden = 32  # 隐藏节点数
n_layers = 1  # 隐藏层数
n_batch = 256
max_iter = 1000
n_display = 1
n_save = 500
lr = 1e-3
# lr = 0.0005

data_path = "./train_180116"
val_path = "./val_180116"

test_path = './test'

check_dir = "./check_GRU"
freeze_dir = check_dir + "/freeze"
logdir = './log_GRU'

test_save = './result'


class Laser_data(object):
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
        return [x_batch, y_batch]


def gru_model(inputs, labels):
    with tf.device("/gpu:0"):
        n_v = n_input
        weights = {
            'input': tf.Variable(tf.random_normal(shape=[n_v, n_hidden], mean=0, stddev=1.0, dtype=tf.float32)),
            'output': tf.Variable(
                tf.random_normal(shape=[n_hidden * 2, n_output], mean=0, stddev=1.0, dtype=tf.float32))}
        biases = {'input': tf.Variable(tf.random_normal(shape=[n_hidden])),
                  'output': tf.Variable(tf.random_normal(shape=[n_output]))}

        print("5:", inputs)
        x = tf.transpose(inputs, [1, 0, 2])  # [0,1,2]->[1,0,2]
        y = tf.transpose(labels, [1, 0, 2])
        print("6:", x)
        print("7:", y)
        x = tf.reshape(x, [-1, n_v])
        print("8:", x)
        x = tf.matmul(x, weights['input']) + biases['input']
        print("9:", x)
        x = tf.split(x, n_step)
        print("10:", x)
        gru_cell = tf.contrib.rnn.GRUCell(n_hidden*2)
        print("11:", gru_cell)
        drop_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, 0.6)
        print("12:", drop_cell)
        mgru_cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * n_layers)
        print("13:", mgru_cell)
        outputs, _ = tf.contrib.rnn.static_rnn(mgru_cell, x, dtype=tf.float32)
        # outputs = tf.concat([outputs, outputs], 2)
        print("14:", outputs)
        out = tf.reshape(outputs, shape=[-1, 2 * n_hidden])
        print("15:", out)
        pred = tf.matmul(out, weights['output']) + biases['output']
        print("16:", pred)
        y = tf.reshape(y, shape=[-1, n_output])
        print("17:", y)
        correct_pred = tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), 'float')
        accuracy = tf.reduce_mean(correct_pred)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=-1)
        cost = tf.reduce_mean(cost, 0)
        pred = tf.reshape(tf.nn.softmax(pred), shape=[n_step, -1, n_output], name='predict')
    return [pred, cost, accuracy]


data = Laser_data(data_path, val_path)
# data.train_x = np.array(data.train_x)
# data.train_y = np.array(data.train_y)
# print("1:", data.train_x.shape)
# print("2:", data.train_y.shape)
x_in = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_input], name='inputs')
y_in = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_output])
print("3:", x_in.shape)
print("4:", y_in.shape)

loss_train_ph = tf.placeholder(tf.float32)
loss_val_ph = tf.placeholder(tf.float32)
loss_train_op = tf.summary.scalar('train_loss', loss_train_ph)
loss_val_op = tf.summary.scalar('val_loss', loss_val_ph)

[pred, loss, acc] = gru_model(x_in, y_in)


def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    train_writer = tf.summary.FileWriter(logdir + '/train')
    val_writer = tf.summary.FileWriter(logdir + '/val')
    acc_writer = tf.summary.FileWriter(logdir + '/acc')
    with tf.device("/gpu:0"):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        cpkt = tf.train.latest_checkpoint(check_dir)
        if cpkt:
            print('restore' + cpkt)
            saver.restore(sess, cpkt)
        tf.train.write_graph(sess.graph_def, freeze_dir, "laser_monitor-model.pbtxt", as_text=True)

        feed_val = {x_in: data.val_x, y_in: data.val_y}
        for i in range(max_iter + 1):
            [x_batch, y_batch] = data.next_batch(n_batch)
            feed = {x_in: x_batch, y_in: y_batch}
            if i % n_display == 0:
                train_loss = sess.run(loss, feed_dict=feed)
                val_loss, val_acc = sess.run([loss, acc], feed_dict=feed_val)
                t_loss, v_loss = sess.run([loss_train_op, loss_val_op],
                                          feed_dict={loss_train_ph: train_loss, loss_val_ph: val_loss})
                train_writer.add_summary(t_loss, i)
                val_writer.add_summary(v_loss, i)
                sum = tf.Summary(value=[tf.Summary.Value(tag="acc", simple_value=val_acc)])
                acc_writer.add_summary(sum, i)
                print('iter {}  train loss : {} , val loss: {} , val_accuracy: {}'.format(i, train_loss,
                                                                                          val_loss,
                                                                                          val_acc))
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


train()
