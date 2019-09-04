# -*-coding:utf-8 -*-
# @Author:king
# @time:2019/7/15 15:03
# @File:mnistgru.py
# @Software:PyCharm

import tensorflow as tf
import numpy as np

'''
构建多层单向RNN网络对MNIST数据集分类
'''
'''
MNIST数据集一个样本长度为28 x 28 
我们可以把一个样本分成28个时间段，每段内容是28个值，然后送入LSTM或者GRU网络
我们设置隐藏层的节点数为128
'''


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


def mnist_rnn_classfication(flag):
    '''
    对MNIST进行分类

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
    from tensorflow.examples.tutorials.mnist import input_data

    # mnist是一个轻量级的类，它以numpy数组的形式存储着训练，校验，测试数据集  one_hot表示输出二值化后的10维
    mnist_data_folder = "/data"
    mnist = input_data.read_data_sets(mnist_data_folder, one_hot=True)

    print(type(mnist))  # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

    print('Training data shape:', mnist.train.images.shape)  # Training data shape: (55000, 784)
    print('Test data shape:', mnist.test.images.shape)  # Test data shape: (10000, 784)
    print('Validation data shape:', mnist.validation.images.shape)  # Validation data shape: (5000, 784)
    print('Training label shape:', mnist.train.labels.shape)  # Training label shape: (55000, 10)

    '''
    2 定义参数，以及网络结构
    '''
    n_input = 28  # LSTM单元输入节点的个数
    n_steps = 28  # 序列长度
    n_hidden = 128  # LSTM单元输出节点个数(即隐藏层个数)
    n_classes = 10  # 类别
    batch_size = 128  # 小批量大小
    training_step = 1000  # 迭代次数
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

    print('hidden:', hiddens[-1].shape)  # (128,128)

    # 取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=n_classes, activation_fn=tf.nn.softmax)

    '''
    3 设置对数似然损失函数
    '''
    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(output), axis=1))

    '''
    4 求解
    '''
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 预测结果评估
    # tf.argmax(output,1)  按行统计最大值得索引
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(input_y, 1))  # 返回一个数组 表示统计预测正确或者错误
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    # 创建list 保存每一迭代的结果
    test_accuracy_list = []
    test_cost_list = []

    with tf.Session() as sess:
        # 使用会话执行图
        sess.run(tf.global_variables_initializer())  # 初始化变量

        # 开始迭代 使用Adam优化的随机梯度下降法
        for i in range(training_step):
            x_batch, y_batch = mnist.train.next_batch(batch_size=batch_size)
            # Reshape data to get 28 seq of 28 elements
            print(x_batch.shape)
            x_batch = x_batch.reshape([-1, n_steps, n_input])
            print(x_batch.shape)

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
            x_batch, y_batch = mnist.test.next_batch(batch_size=50)
            # Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1, n_steps, n_input])
            test_accuracy, test_cost = sess.run([accuracy, cost], feed_dict={input_x: x_batch, input_y: y_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost)
            if (i + 1) % 20 == 0:
                print('Step {0}:Test set accuracy {1},cost {2}.'.format(i + 1, test_accuracy, test_cost))
        print('Test accuracy:', np.mean(test_accuracy_list))


if __name__ == '__main__':
    mnist_rnn_classfication(1)  # 1：多层静态LSTM
    mnist_rnn_classfication(2)  # 2：多层静态gru
    mnist_rnn_classfication(3)  # 3: 多层静态LSTM和gru混合网络：
    mnist_rnn_classfication(4)  # 4：多层动态LSTM
    mnist_rnn_classfication(5)  # 5：多层动态gru
    mnist_rnn_classfication(6)  # 3: 多层动态LSTM和gru混合网络：