# -*- coding: utf-8 -*-
import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent #此節點的父節點與子節點
        self._children = {}  
        self._n_visits = 0 #拜訪次數
        self._Q = 0 
        self._u = 0
        self._P = prior_p #預設是1.0 因為最初初始化時最初的節點"選擇率"是100%

    def expand(self, action_priors):
        #這邊如果(print self._children)的時候通常會一直是空的 這是因為我們目前這個節點下面肯定都是沒有節點的 所以才要expend 詳見MCTS原理
        for action, prob in action_priors:
            if action not in self._children:  #使用策略端的輸出prob創建拜訪64個節點
                self._children[action] = TreeNode(self, prob)


    def select(self, c_puct):
        #max函數:首先key是一個函數會對self._children.items()進行處理
        #1.一開始_children為空的 呼叫items會返回一個空的字典 而正常_children是表示父節點的"下一排"所有可能的子節點 因為我們要選擇哪一個為下一個子節
        #2.key是一個function 會套用到_children.items()上 呼叫_children.items的get_value
        #3.就會得到這個_children.items最新的Q值
        #4.運行max選出Q值最大的 選擇他 return
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        #alphago論文第5頁 新的Q價值的算法
        

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value) #因為一個節點的父節點是其對手，故要更新負值
        self.update(leaf_value)

    def get_value(self, c_puct):
        #計算一個節點的價值
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits)) #np.sqrt返回平方根
        return self._Q + self._u

    def is_leaf(self):
        #確認是否是根節點
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0) #創立最初的樹節點
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root #得到最初的節點
        while(1):
            if node.is_leaf(): #如果到達最後的葉子則break (偵測當前的node是不是下面沒有children了)
              #那為什麼下面還可能會有children呢? 因為每次都是從"頭"開始搜索 所以有可能選擇到上次選擇的move 所以要執行select步驟必須直到選到葉尾的棋子
                break
            #Select
            action, node = node.select(self._c_puct)
            state.do_move(action)
        action_probs, leaf_value = self._policy(state) #從神經網路策略端取得目前合法落子點的動作機率、價值端取得目前局面價值
        end,winner = state.game_end()
        #根據模擬棋盤是否結束來決定如何更新 沒結束的話就expand後用價值端更新
        #結束的話就使用0或1(根據我方輸贏)
        if not end:
            node.expand(action_probs)
        else: #更新
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        node.update_recursive(-leaf_value) #如果還沒結束 從此節點直到樹頭用價值端的輸出更新

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout): #執行400次
            state_copy = copy.deepcopy(state) #將state進行深拷貝 與淺拷貝的差別詳見網路
            #而之所以用copy是因為MCTS是模擬搜尋 環境要跟我們正在玩的環境分開
            self._playout(state_copy)
        #取出樹頭之下第一排的動作 因為第一排的動作就是現實棋盤當前的合法落子點
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        #取出
        acts, visits = zip(*act_visits)
        #機率根據節點的拜訪次數決定 因為原本節點的P並不能用(在算法中不會更新)
        #temp為探索手段 訓練時設為1 對打時設為0(但除以0會錯誤故改使用預設值temp=1e-3
        '''
        根據alphago論文 第24頁
        #using an infinitesimal temperature τ → 0 (i.e. we deterministically select the move with maximum visit count, to give the
strongest possible play).
    使用一個接近0的無窮小的數值
        但在這使用1e-3 代表1/0.001=1000 所以當兩個節點拜訪次數分別為10跟11時會擴增到1000以及1100 根據softmax算法(根據最小值與最大值的間距做比例 將所有值壓縮至1~0)
        11節點的機率會較大(比temp=1時還大) 也就達成了我們對打時想使用最強的選擇的目的
        '''
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        由於MCTS是使用你上一步的動作下去做MCTS搜尋 這代表說我們根節點要從上一步開始 也就是說下次MCTS搜尋要從你當前選擇的動作開始下去搜尋
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=400, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        
    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = list(board.availables) #得到目前所有可用的格子
        move_probs = np.zeros(board.width*board.height) #生成棋盤格子數的一個全是0的一維陣列
        if len(sensible_moves) > 0: #如果還有格子能下
            acts, probs = self.mcts.get_move_probs(board, temp) #MCTS搜索
            move_probs[list(acts)] = probs #probs由於是神經網路策略端給出的，見policy_value_fn函式，會給目前所有合法落子點的機率
            if self._is_selfplay:
                #添加狄利克雷躁點於選擇動作的機率中作為探索手段
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))) #dirichlet機率分布 簡單來說就是我們把probs全部拿出來 通通*0.3 接著np.random.dirichlet會找出他最趨近於哪個值 比如硬幣最可能是0.5
#所以0.3就是決定了 我們究竟要讓他找出多大多小的分布 決定了這0.25%究竟會有多少機率 詳見https://zhuanlan.zhihu.com/p/24555092
                )
                #將樹的頭指向合法落子點的位子 下次從它開始搜尋
                self.mcts.update_with_move(move)
            else:
                #不是訓練時就直接根據機率選擇機率最大的為動作 因為我們不需要探索也就不用添加
                move = np.random.choice(acts, p=probs)
                #樹也是每次都是從頭開始建立 變成主要依靠神經網路來去做決策 而不是使用上一次的樹
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
