# -*-coding:utf-8 -*-
# @Author:king
# @time:2019/7/15 11:21
# @File:mutligru.py
# @Software:PyCharm

import tensorflow as tf
import numpy as np
import os


class Laser_data(object):
    def __init__(self, train_path):
        self.sample_names = []
        self.train_x = []
        self.train_y = []
        self.val_x = []
        self.val_y = []
        [self.train_x, self.train_y] = self.read_data(train_path)
        # [self.val_x, self.val_y] = self.read_data(val_path)

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
                    # label = []
                    # if cls == 1.0:
                    #     label = [0.0, 1.0]
                    # else:
                    #     label = [1.0, 0.0]
                    # dist_label.append(label)
                    dist_label.append(cls)

                x.append(dist)
                labels.append(dist_label)

        return [x, labels]

    def next_batch(self, batch_size):
        arr = np.arange(len(self.train_x))
        np.random.shuffle(arr)
        indexs = arr[:batch_size]
        x_batch = [self.train_x[i] for i in indexs]
        y_batch = [self.train_y[i] for i in indexs]
        return [x_batch, y_batch]


def multi_layer_static_lstm(input_x, n_steps, n_hidden):
    '''
    返回静态多层LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''

    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.LSTMCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.contrib.rnn.static_rnn(cell=mcell, inputs=input_x1, dtype=tf.float32)

    return hiddens, states


def multi_layer_static_gru(input_x, n_steps, n_hidden):
    '''
    返回静态多层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.GRUCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.contrib.rnn.static_rnn(cell=mcell, inputs=input_x1, dtype=tf.float32)

    return hiddens, states


def multi_layer_static_mix(input_x, n_steps, n_hidden):
    '''
    返回静态多层GRU和LSTM混合单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

    # 可以看做2个隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden * 2)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_hidden)

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell, gru_cell])

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.contrib.rnn.static_rnn(cell=mcell, inputs=input_x1, dtype=tf.float32)

    return hiddens, states


def multi_layer_dynamic_lstm(input_x, n_steps, n_hidden):
    '''
    返回动态多层LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量  形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.LSTMCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


def multi_layer_dynamic_gru(input_x, n_steps, n_hidden):
    '''
    返回动态多层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.GRUCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


def multi_layer_dynamic_mix(input_x, n_steps, n_hidden):
    '''
    返回动态多层GRU和LSTM混合单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 可以看做2个隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden * 2)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_hidden)

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell, gru_cell])

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


def laser_rnn_detection(flag):
    '''

        arg:
            flags:表示构建的RNN结构是哪种
                1：多层静态LSTM
                2: 多层静态GRU
                3：多层静态LSTM和GRU混合
                4：多层动态LSTM
                5: 多层动态GRU
                6: 多层动态LSTM和GRU混合
    '''
    '''
    1. 导入数据集
    '''
    tf.reset_default_graph()
    data_path = "./train_180116"
    # val_path = "./val_180116"
    data = Laser_data(data_path)
    '''
    2 定义参数，以及网络结构
    '''
    n_input = 1  # LSTM单元输入节点的个数
    n_steps = 361  # 序列长度
    n_hidden = 32  # LSTM单元输出节点个数(即隐藏层个数)
    n_classes = 2  # 类别
    batch_size = 256  # 小批量大小
    training_step = 5000  # 迭代次数
    display_step = 200  # 显示步数
    learning_rate = 1e-4  # 学习率

    # 定义占位符
    # batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入LSTM网络
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input])
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

    # 可以看做隐藏层
    if flag == 1:
        print('多层静态LSTM网络：')
        hiddens, states = multi_layer_static_lstm(input_x, n_steps, n_hidden)
    elif flag == 2:
        print('多层静态gru网络：')
        hiddens, states = multi_layer_static_gru(input_x, n_steps, n_hidden)
    elif flag == 3:
        print('多层静态LSTM和gru混合网络：')
        hiddens, states = multi_layer_static_mix(input_x, n_steps, n_hidden)
    elif flag == 4:
        print('多层动态LSTM网络：')
        hiddens, states = multi_layer_dynamic_lstm(input_x, n_steps, n_hidden)
    elif flag == 5:
        print('多层动态gru网络：')
        hiddens, states = multi_layer_dynamic_gru(input_x, n_steps, n_hidden)
    elif flag == 6:
        print('多层动态LSTM和gru混合网络：')
        hiddens, states = multi_layer_dynamic_mix(input_x, n_steps, n_hidden)

    print('hidden:', hiddens[-1].shape)

    # 取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    # output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=n_classes,
    #                                            activation_fn=tf.nn.softmax)
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, 0.6)
    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=[gru_cell], state_is_tuple=True)

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    output, _ = tf.contrib.rnn.static_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)
    print(output)
    # output = tf.reshape(output, shape=[-1, 2 * n_hidden])
    # print(output)
    '''
    3 设置对数似然损失函数
    '''
    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    # cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(output), axis=1))
    # cost = tf.nn.softmax_cross_entropy_with_logits(logits=input_y, labels=input_y, dim=-1)
    # cost = tf.reduce_mean(cost, 0)
    '''
    4 求解
    '''
    with tf.device("/gpu:0"):
        weights = {'input': tf.Variable(tf.random_normal(shape=[n_input, n_hidden], mean=0, stddev=1.0, dtype=tf.float32)),
                   'output': tf.Variable(
                       tf.random_normal(shape=[n_hidden * 2, n_classes], mean=0, stddev=1.0, dtype=tf.float32))}
        biases = {'input': tf.Variable(tf.random_normal(shape=[n_hidden])),
                  'output': tf.Variable(tf.random_normal(shape=[n_classes]))}
        out = tf.reshape(output, shape=[-1, 2 * n_hidden])
        pred = tf.matmul(out, weights['output']) + biases['output']
        input_y = tf.transpose(input_y, [1, 0])
        input_y = tf.reshape(input_y, [-1, n_classes])
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=input_y, dim=-1)
        cost = tf.reduce_mean(cost, 0)
        train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 预测结果评估
    # tf.argmax(output,1)  按行统计最大值得索引
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(input_y, 1))  # 返回一个数组 表示统计预测正确或者错误
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    # 创建list 保存每一迭代的结果
    test_accuracy_list = []
    test_cost_list = []

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 使用会话执行图
        sess.run(tf.global_variables_initializer())  # 初始化变量

        # 开始迭代 使用Adam优化的随机梯度下降法
        for i in range(training_step):
            x_batch, y_batch = data.next_batch(batch_size=batch_size)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            y_batch = np.reshape(y_batch, [-1, 2])
            # print(x_batch.shape, y_batch.shape)
            # Reshape data to get 28 seq of 28 elements
            # x_batch = np.array(x_batch)
            # print(x_batch.shape)
            # x_batch = x_batch.reshape([-1, n_steps, n_input])
            # print(x_batch.shape)

            # 开始训练
            train.run(feed_dict={input_x: x_batch, input_y: y_batch})
            if (i + 1) % display_step == 0:
                # 输出训练集准确率
                training_accuracy, training_cost = sess.run([accuracy, cost],
                                                            feed_dict={input_x: x_batch, input_y: y_batch})
                print('Step {0}:Training set accuracy {1},cost {2}.'.format(i + 1, training_accuracy, training_cost))

        # 全部训练完成做测试  分成200次，一次测试50个样本
        # 输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试
        for i in range(200):
            x_batch, y_batch = data.next_batch(batch_size=50)
            # Reshape data to get 28 seq of 28 elements
            # x_batch = np.array(x_batch)
            # x_batch = x_batch.reshape([-1, n_steps, n_input])
            # x_batch = tf.reshape(x_batch, [-1, n_input])
            test_accuracy, test_cost = sess.run([accuracy, cost], feed_dict={input_x: x_batch, input_y: y_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost)
            if (i + 1) % 20 == 0:
                print('Step {0}:Test set accuracy {1},cost {2}.'.format(i + 1, test_accuracy, test_cost))
        print('Test accuracy:', np.mean(test_accuracy_list))


if __name__ == '__main__':
    laser_rnn_detection(1)  # 1：多层静态LSTM
    laser_rnn_detection(2)  # 2：多层静态gru
    laser_rnn_detection(3)  # 3: 多层静态LSTM和gru混合网络：
    laser_rnn_detection(4)  # 4：多层动态LSTM
    laser_rnn_detection(5)  # 5：多层动态gru
    laser_rnn_detection(6)  # 3: 多层动态LSTM和gru混合网络：
