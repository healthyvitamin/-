# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        if model_file is not None:
            tf.reset_default_graph()
        
        self.board_width = board_width
        self.board_height = board_height
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1]) #由於
        #我們下面CNN用data_format="channels_last 他會限定資料型態是(batch, height, width, channels)
        #所以要transpose 將維度換位子
        # 2. 三層CNN
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 3-1 策略端 輸出action prob
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        #因為前面CNN改了形狀 這邊把它改回來
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 全連接層 用softmax得到每個棋盤位子action的概率 
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 價值端 輸出當前局面的價值
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape( #同策略端 要reshape回來
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        #tanh得到-1~1之間的值 代表當前局面的價值
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        #定義loss fc
        #此為作者在selfplay中儲存的一個數據 即贏的是當前玩家的話便全是1 不是則0
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 3-1. 價值網路的loss fc  用mean_squared_error表示
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        
        #此為在selfplay中儲存的數據 即MCTS輸出的action prob 
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        # 3-2. 策略網路的loss fc  multiply矩陣各自的數相乘、reduce_sum(x,1) 按行求和 見google 
        #reduce_mean求所有值平均、negative負號
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2正則化
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables() #取得目前所有需要訓練的變量
        # tf.nn.l2_loss(v)輸出sum(v ** 2) / 2 見https://blog.csdn.net/yangfengling1023/article/details/82910536
        #tf.add_n把值加總 見https://www.tensorflow.org/api_docs/python/tf/math/add_n
        #並且變數沒有bias名字的才需要套用 
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()]) 
        # 3-4 將以上三個組成最後的loss fc
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # 優化器
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
        #決定占用GPU的記憶體容量
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.2
            )
        )
        self.session = tf.Session(config=config)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/", self.session.graph)
        # 計算entropy 觀察用
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))
        #tf.exp輸出 e^x = ?? 例如e^0 = 1  
        self.entropy_summ = tf.summary.scalar('entropy',self.entropy)
        # 初始化
        if model_file is None:
            init = tf.global_variables_initializer()
            self.session.run(init)

        # 儲存model
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.saver = tf.train.import_meta_graph(model_file+'.meta')
            self.restore_model(model_file)


    def policy_value(self, state_batch):
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        legal_positions = list(board.availables) #從game.py得到棋盤的點
        #取得當前棋盤狀態 
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        #np.ascontiguousarray 可返回跟輸入一樣的值 但是是C style
        act_probs, value = self.policy_value(current_state) #把棋盤點輸入到神經網路得到action的概率、目前局面的value
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        #將"action的概率"對應到合法落子點上 其餘的不要 這樣就完成了output
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        winner_batch = np.reshape(winner_batch, (-1, 1)) #將winner_batch reshape成(-1,1)的形狀 因為labels我們形狀設定成[-1,1]
        loss, entropy, _ = self.session.run(  #開始更新網路、查看loss、entropy
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
