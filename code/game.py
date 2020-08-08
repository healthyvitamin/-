# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8)) #get拿取傳進來的width 如果沒有 回傳默認值8
        self.height = int(kwargs.get('height', 8))
        self.states = {}
        self.players = [1, 2]  # player1 and player2
        self.moved=[]

    def init_board(self, start_player=1):
        self.current_player = self.players[start_player] #設定先手玩家
        self.states = {}
        for move in [27,36]:
            self.states[move] = self.players[1] #黑棋
        for move in [28,35]:
            self.states[move] = self.players[0] #白旗
        self.chessman_number={'white':0,'black':0} #不需要把一開始棋盤的數量寫進去 因為後面偵測會全部偵測到 寫進去會多2+2顆棋子
        self.moved = [27,28,36,35] #當前以用過的動作
        self.last_move = -1
        #用於解決掃描時行列的問題(對掃描起始點進行分類用)
        self.upperleft = {'0':[7,0],'1':[6,1],'2':[5,2],"3":[4,3],"4":[3,4],"5":[2,5],"6":[1,6],"7":[0,7]}
        self.scanning_for_availables()  #初始化棋盤後 要先找出start_player能夠下的點 注意
    def move_to_location(self, move):
        #動作轉行列
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location): #行列轉動作
        if len(location) != 2: #如果輸入的不是兩個數字 錯誤 -1
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w #都對的話return move
        if move not in range(self.width * self.height): #如果輸入的數字大於棋盤 錯誤 -1
            return -1
        return move

    def current_state(self):
        """ 也就是返回我們當前正確的狀態 最後會送到train get_equi_data 進行最  
        後加工，每次self_play下完一步就會蒐集4種棋盤狀態
        1.我方的棋盤狀態(即一個陣列，我方棋子位子1，其餘的都0)
        2.對方的，同上
        3.最後一步的位子
        4.當前下這一步的是我方則全1的陣列，是對方則全-1
        變成輸入進神經網路的states
        
        這是為了決定蒐集的資訊
        所以只蒐集前面2個也能夠訓練，但成效好不好就要case by case了
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player] #文章講到的 當當前對手是自己
            #或是對方時 要換一下
            move_oppo = moves[players != self.current_player]
            #這裡我們將每一次
            #我方與對方的棋盤狀態
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            #最後一步的位子
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        #根據當前下棋的是我方還是對方的一個全1或全-1圖形
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  
        return square_state[:, ::-1, :] 

    def do_move(self, move):
        self.states[move] = self.current_player
        self.moved.append(move) #儲存已經用過的位子
        for n in self.availables[move]:#反轉棋子 將字典儲存的值拿出來
            self.states[n] = self.current_player

        self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
                )

        self.scanning_for_availables() #重新掃描新的對於下個對手可下的點

        if not len((self.availables)): #黑白棋規則 假如你放下棋子 對方沒地方可下 則你可以再下一次
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
                )
            self.scanning_for_availables() #所以還要再掃描一次回你的
        self.last_move = move


    def game_end(self): 
        states = self.states
        if not len(self.availables): #如果沒地方可下了 且對方也沒地方可下 則結束
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
                )
            self.scanning_for_availables()
            if not len(self.availables):
                for m in self.moved:
                    if states.get(m,-1)==1:
                        self.chessman_number['white']+=1
                    elif states.get(m,-1)==2:
                        self.chessman_number['black']+=1
                    else:
                        print("最後掃描到空白處 還沒結束! moved卻到達64 位子已下完 ")
                if self.chessman_number['white'] > self.chessman_number['black']:
                    #白棋獲勝

                    return True,self.players[0]
                elif self.chessman_number['black'] > self.chessman_number['white']:
                    #print("黑棋獲勝")
                    return True,self.players[1]
                else: # true -1代表平手
                    return True , -1
            else: #false -1 代表還沒結束
                return False,-1
        else:
            return False,-1

    def scanning_for_availables(self):
        self.availables={}
        store_opponent = []

        #2019/3/20 另一種方法 利用餘數可以知道這個數字在第幾列 整除可以知道在第幾行 !!!
        #接著例如你在第三列第五行 那麼往右搜尋就只需要搜索到 7-5 =2 2格即可!! 這樣就不需要管邊疆問題了
        #2019/3/22 解決掃描 接下來解決用字典儲存偵測到的黑子其的move問題
        #再來是怎麼翻轉

        for m in self.moved: #看每個已做過的動作
            w = m // self.width #取整數 看動作在第幾列
            h = m % self.width #取餘數  看動作在第幾行 以設定究竟要檢查幾格 解決超出邊疆的問題
            h_class=0
            w_class=0
            right_upperleft=0

            if h >= w :
                h_class = 1
            elif h < w :
                w_class = 1
            if w > self.upperleft[str(h)][0] :
                right_upperleft = 1
            player= self.states[m] #查看這個動作在state上是上面哪一個玩家的符號
            if player == self.current_player: #如果是當前玩家的旗子 則開始掃描該棋子以找出可下的點
                #往右檢查
                for i in range(m+1,m+(7-h+1)):

                    #首先檢查這顆是自己的棋以及檢查第一顆是不是空白處 都代表不需要繼續掃描 跳出迴圈
                    if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+1):
                        break
                    elif self.states.get(i,-1) == -1: #接下來繼續掃描 如果碰到空白處 此處即是可下的點(因有黑棋間隔著)
                        if self.availables.__contains__(i):#此句在確認是否重複掃描 沒有的話availables[i]會被蓋掉 如果重複那麼將
                            self.availables[i].extend(store_opponent) #以此可下的點為首 與沿途掃描到的黑棋 存成字典
                        else:
                            self.availables[i]=store_opponent
                        break
                    else:
                        #檢查這顆是不是自己的棋，前面已檢查過，故這邊可刪除
                        if (self.states.get(i,-1) == self.current_player):
                            break
                        else:
                            #如果是對手棋 就把對手的棋子加入到store_opponent
                            store_opponent.append(i)
                store_opponent=[]
                #往右上檢查
                if h_class: #如果屬於右下半部
                    for i in range(m+9,m+9*(7-h+1),self.width+1):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+9):
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):

                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]
                             #往左下檢查 同樣可以利用分類好的左上右下半部  這邊右下半部使用w計算
                    for i in range(m-9,m-9*(w+1),-(self.width+1)):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-9) :
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]

                if w_class:    #如果屬於左上半部
                    for i in range(m+9,m+9*(7-w+1),self.width+1):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+9):
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                             #往左下檢查 這邊是左上半部使用h計算
                    store_opponent=[]
                    for i in range(m-9,m-9*(h+1),-(self.width+1)):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-9) :
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]

                #往上檢查
                for i in range(m+8,m+8*(7-w+1),self.width):
                    if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+8) :
                        break
                    elif self.states.get(i,-1) == -1:
                        if self.availables.__contains__(i):
                            self.availables[i].extend(store_opponent)
                        else:
                            self.availables[i]=store_opponent
                        break
                    else:
                        if (self.states.get(i,-1) == self.current_player):
                            break
                        else:
                            #如果是對手棋 就把對手的棋子加入到
                            store_opponent.append(i)
                store_opponent=[]
                #往左上檢查
                if right_upperleft: #如果屬於右上半部
                    for i in range(m+7,m+7*(7-w+1),self.width-1):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+7):
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]
                    #我們這邊是右下檢查 同上面解釋 同樣可以利用分類好的左下右上半部  這邊右上半部使用7-h計算
                    for i in range(m-7,m-7*(7-h+1),-(self.width-1)):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-7) :
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]

                else: #這邊也是左上檢查 但如果是左下半部或中間 可以直接用h得出要偵測幾格
                    for i in range(m+7,m+7*(h+1),self.width-1):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m+7):
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]
                    #我們這邊是右下檢查 這邊左下半部則使用w計算
                    for i in range(m-7,m-7*(w+1),-(self.width-1)):
                        if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-7) :
                            break
                        elif self.states.get(i,-1) == -1:
                            if self.availables.__contains__(i):
                                self.availables[i].extend(store_opponent)
                            else:
                                self.availables[i]=store_opponent
                            break
                        else:
                            if (self.states.get(i,-1) == self.current_player):
                                break
                            else:
                                #如果是對手棋 就把對手的棋子加入到
                                store_opponent.append(i)
                    store_opponent=[]

                #往左邊檢查    可直接用h
                for i in range(m-1,m-h-1,-1):
                    if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-1) :
                        break
                    elif self.states.get(i,-1) == -1:
                        if self.availables.__contains__(i):
                            self.availables[i].extend(store_opponent)
                        else:
                            self.availables[i]=store_opponent
                        break
                    else:
                        if (self.states.get(i,-1) == self.current_player):
                            break
                        else:
                            #如果是對手棋 就把對手的棋子加入到
                            store_opponent.append(i)
                store_opponent=[]


                #往下檢查   可直接用w
                for i in range(m-8,m-8*(w+1),-self.width):
                    if (self.states.get(i,-1) == self.current_player) or (self.states.get(i,-1) == -1 and i == m-8) :
                        break
                    elif self.states.get(i,-1) == -1:
                        if self.availables.__contains__(i):
                            self.availables[i].extend(store_opponent)
                        else:
                            self.availables[i]=store_opponent
                        break
                    else:
                        if (self.states.get(i,-1) == self.current_player):
                            break
                        else:
                            #如果是對手棋 就把對手的棋子加入到
                            store_opponent.append(i)
                store_opponent=[]

    def get_current_player(self):
        return self.current_player

class Game(object):

    def __init__(self, board, **kwargs):
        self.board = board

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, #從MCTSplayer
                                                 temp=temp,
                                                 return_prob=1)

            states.append(self.board.current_state())
            mcts_probs.append(move_probs) #mcts_probs為每一場的機率
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            end , winner = self.board.game_end()
            if end == True:
                
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player() #重置mcts的樹
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
