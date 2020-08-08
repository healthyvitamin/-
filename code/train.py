# -*- coding: utf-8 -*-


from __future__ import print_function
import random
import numpy as np
from collections import deque #collecton是python內建的
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet
import matplotlib.pyplot as plt

class TrainPipeline():
    def __init__(self, init_model=None):
        #設定、創建棋盤
        self.board_width = 8
        self.board_height = 8
        self.board = Board(width=self.board_width,
                           height=self.board_height
                           )
        self.game = Game(self.board)
        #設定training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  #後面根據kl差異自適應調整學習率 以加快速度
        self.temp = 1.0  # 探索參數:溫度 添加於玩家訓練時的get_action
        self.n_playout = 400  #MCTS搜索次數
        self.c_puct = 5 #探索參數:添加於計算節點U值中
        self.buffer_size = 10000  #總共儲存多少data
        self.batch_size = 512  #一次喂進神經網路的資料量
        self.data_buffer = deque(maxlen=self.buffer_size) #deque類似於list 詳見網路 但可自動把字串拆開來儲存，還可從左邊插入新值 以及一堆與list一樣的功能
        self.play_batch_size = 1 #訓練一局蒐集一次資料
        self.epochs = 5  # 每次神經網路訓練五次
        self.game_batch_num = 3000 #總共訓練n局
        if init_model:  #使用已有的Model
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else: #重頭開始訓練
            self.policy_value_net = PolicyValueNet(self.board_width, #創建策略價值網路
                                                   self.board_height)
        #創建MCTSPlayer
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        #plt畫圖用。
        fig = plt.figure()
        self.entropy_list =[]
        self.ax = fig.add_subplot(1,1,1) #在figure裡面生成一個子圖 一行一列 位子在一(因其實可有多個子圖 所以有位子的設定)
        self.ax.scatter(5, 6) #子圖為散點圖 位置在x=5  y=6
        plt.ion()
        plt.show()

    def get_equi_data(self, play_data):
        """
        棋類的鏡像與翻轉特性 可把1個棋盤狀態擴充成8個
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                #旋轉
                equi_state = np.array([np.rot90(s, i) for s in state]) 
                #將MCTS每步所儲存的state逆轉90*i度
                #np.flipud將矩陣上下反轉(之所以上下翻轉是因為棋盤編號方式是從下到上0~63)
                #至於為何動作機率也要做旋轉 因為機率是神經網路策略端給的 見policy_value_fn函式
                #當下棋盤狀態有幾個合法落子點就有幾個機率，並且機率是對應到動作(位子)的，所以這裡為了對應到旋轉後的棋盤的合法落子點 也要做處理
                equi_mcts_prob = np.rot90(np.flipud(  
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                #將mcts_porb reshape成二維然後flipud 旋轉 為何reshape? 因為prob原本只是一個list shape為(36,)
                #flatten可將矩陣、數組等弄平 [[1],[2]]變成[[1,2]]
                #mcts_porb經過這些處理後便能對應到"旋轉"後的棋盤了(翻轉不必)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(), \
                                    winner))
                #翻轉
                #np.fliplr將矩陣水平反轉
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))


        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            #訓練一局得到每步的玩家、data
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            
            play_data = self.get_equi_data(play_data) #反轉資料 所以60個會變成60*8 480個
            self.data_buffer.extend(play_data) #將play_data儲存進我們上面設定的一個queue

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size) #random.sample非重置抽樣
        #從self.data_buffer中抽出的 抽出來的不會放回直到抽出self.batch_size個 所以最後會返回self.batch_size的一個list
        
        state_batch = [data[0] for data in mini_batch] #將每個data的第一個取出 而因為mini_batch每個元素是由states, mcts_probs, winners_z組成
        #所以此代表取出mini_batch所有的state
        
        mcts_probs_batch = [data[1] for data in mini_batch] #同理
        winner_batch = [data[2] for data in mini_batch] #同理
        old_probs, old_v = self.policy_value_net.policy_value(state_batch) #從還沒更新的PolicyValueNet中得到這個state的動作機率、價值
        for i in range(self.epochs): #更新epochs次
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate)
            #同old，取得更新後的值
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            
        #查看價值端的變異數 詳見網路 用以觀察價值端的輸出情況 在經過長時間訓練後一直是0的話代表訓練可能有問題(因為輸出都是同一個值 代表收斂的情況不好)
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) / #var 也就是機率中的變異數
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print((
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size) #生成self-play data
                print("第i:{}局, 此局的步數:{}".format(
                        i+1, self.episode_len))
                
                #只要儲存的資料量大於batch_size就更新神經網路、plt圖 (所以第二局之後都會Plt)
                if len(self.data_buffer) > self.batch_size:
                    #更新神經網路 取得更新後的loss
                    loss, entropy = self.policy_update()
                    
                    self.entropy_list.append(entropy)
                    #plot圖 每一次都重畫圖 內容為entropy_list
                    self.ax.plot(np.arange(len(self.entropy_list)) , self.entropy_list)
                    #一定要給等待時間 否則無法畫圖
                    plt.pause(1)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
