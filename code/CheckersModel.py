# -*- coding: utf-8 -*-
"""
Created on Fri May  3 03:16:16 2019

@author: amd
"""

import numpy
import sklearn.cluster
from CVBackwardCompat import cv2

#import ColorUtils
import sys
import win32gui,win32con
from PIL import ImageGrab

import game
from policy_value_net_tensorflow import PolicyValueNet
from mcts_alphaZero import MCTSPlayer

ROTATION_0 = 0
ROTATION_CCW_90 = 1
ROTATION_180 = 2
ROTATION_CCW_270 = 3

DIRECTION_UP = 0
DIRECTION_LEFT = 1
DIRECTION_DOWN = 2
DIRECTION_RIGHT = 3

SQUARE_STATUS_UNKNOWN = -1
SQUARE_STATUS_EMPTY = 0
SQUARE_STATUS_exist = 22

class CheckersModel(object):

#P172--------------------------------------------------------------------------
#property功能:這裡只用@property getter 例如呼叫sceneSize便可取得_sceneSize參數
    @property
    def sceneSize(self):
        return self._sceneSize


    @property
    def scene(self):
        return self._scene



    @property
    def boardSize(self):
        return self._boardSize


    @property
    def white_scene(self):
        return self._white_scene

    @property
    def black_scene(self):
        return self._black_scene
#p173--------------------------------------------------------------------------

    @property
    def white_squareStatuses(self):
        return self._white_squareStatuses
    @property
    def black_squareStatuses(self):
        return self._black_squareStatuses

#p174--------------------------------------------------------------------------
    def __init__(self, patternSize=(7, 7),sceneSize=(800, 600)):
        self.emptyFreqThreshold = 0.3
        self.playerDistThreshold = 0.4
#p175--------------------------------------------------------------------------
#_scene用於顯示現實棋盤(全彩)
#_white_scene、_black_scene則是二值化後的棋盤 為了偵測棋子用
        self._scene = None #儲存棋盤的影像
        self._white_scene = None
        self._black_scene = None

        self._patternSize = patternSize #此為棋盤內部的線的數量 7*7  所以總共會有8*8格
        self._numCorners = patternSize[0] * patternSize[1]
        self._squareFreqs = numpy.empty(  #之後要分類方格內兩個主要色彩的頻率 所以我們先建立兩個空的NUMPY
                (patternSize[1] + 1, patternSize[0] + 1), #因為patternSize[1]是7 所以要+1
                numpy.float32)

        self._white_squareStatuses = numpy.empty( #此為分類結果的numpy陣列 一樣先建立
                (patternSize[1] + 1, patternSize[0] + 1),
                numpy.int8)
        self._white_squareStatuses.fill(SQUARE_STATUS_UNKNOWN) #並且因為還沒偵測 用fill把全部值改成-1

        self._black_squareStatuses = numpy.empty( #此為分類結果的numpy陣列 一樣先建立
                (patternSize[1] + 1, patternSize[0] + 1),
                numpy.int8)
        self._black_squareStatuses.fill(SQUARE_STATUS_UNKNOWN) #並且因為還沒偵測 用fill把全部值改成-1
#p176--------------------------------------------------------------------------
        self._clusterer = sklearn.cluster.MiniBatchKMeans(2) #K-means群集分類方法 2代表我們想將輸入的資料分成2個群集用以偵測兩種顏色(將黑白棋二值化後使用)
        
        #一開始先用win32gui尋找遊戲視窗 以顯示 後續更新會呼叫_updateScene函式
        self.hwnd = win32gui.FindWindow(None ,'BlueStacks')
        try :
            """
            #SetWindowPos將視窗固定在某個位子 參數設定見
            https://www.cnblogs.com/qq78292959/archive/2013/04/09/3010059.html
            https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowpos
            https://bbs.csdn.net/topics/390621834
            #這裡使用SWP_NOSIZE維持當前尺寸 HWND_TOPMOST(也就是第二個參數設定為-1)置於所有視窗最頂層
            """
            win32gui.SetWindowPos(self.hwnd, -1, 10, 10, 0, 0, win32con.SWP_NOSIZE)
            #取得視窗大小
            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
            #擷取視窗一部份(這邊要自己手動調整 只要棋盤就好 因為我們要根據畫面去切割出8*8方格)
            img =  ImageGrab.grab(bbox = (left+5, top+180, right-10, bot-230))
        except :
            print("找不到視窗")
            sys.exit()
        #取得擷取的大小
        self._sceneSize = img.size
        w, h = self._sceneSize
        w = int(w) #由於numpy.empty不支援使用float當作輸入(因為輸入只是要形狀) 故要轉型成int
        h = int(h)
        self._squareWidth = min(w, h) // (max(patternSize) + 1) #1.從寬高中選擇較短的計算棋盤寬度 得到一個方塊的寬度
        self._squareArea = self._squareWidth ** 2  #2.所以一個方塊正方型面積是寬度平方
        self._boardSize = (                     #3.棋盤大小每一邊則是8*方塊寬度
            (patternSize[0] + 1) * self._squareWidth,
            (patternSize[1] + 1) * self._squareWidth
        )
#------------------------------------------------------------------------------
        #傳入game.py的board建立棋盤
        self.board = game.Board(width=patternSize[0]+1, height=patternSize[1]+1)
        #初始化棋盤、設定我們要偵測哪一方(AI要扮演的棋子) 0是白棋 1是黑棋
        self.board.init_board(start_player=0)
        #決定神經網路要使用的model、創建神經網路、AI(MCTSPlayer
        model_file = '20190514current_policy.model'
        best_policy = PolicyValueNet(patternSize[0]+1, patternSize[1]+1, model_file = model_file)
        self.mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

#177~178------------------------------------------------------------------------
#在Checkers.py 副執行緒運行的_runCaptureLoop會呼叫這個函式
    def update(self,drawSquareStatuses=True): #drawSquareStatuses分類結果是否要顯示在棋盤俯視圖中

        if not self._updateScene(): #1.呼叫_updateScene方法看是否成功讀取到圖片
            return False  # Failure
        self._updateBoard() #2.處理影像辨識棋子
        w,h=self.scanningToAI() #3.取得棋盤狀態接著送至AI取得AI的動作

        #4.繪製於圖中
        w=7-w
        self._drawSquareStatus(h,w,AI=True) #由於我們畫字是根據XY座標 而不是行列所以假設動作20 w=7-2=5 h=4 h才是X座標 w是Y座標 所以這邊要交換
        if drawSquareStatuses:  #_drawSquareStatus fc將分類結果顯示在指定方格裡
            for i in range(self._patternSize[0] + 1):
                for j in range(self._patternSize[1] + 1):
                    self._drawSquareStatus(i, j)
        #5.回傳true 讓主執行緒更新影像
        return True  # Success

#179----------------------------------------------------------------------------

    def _updateScene(self):
        #同P176 擷取視窗取得三個影像
        try :
            #SetWindowPos參數設定同上面init時
            win32gui.SetWindowPos(self.hwnd, -1, 10, 10, 0, 0, win32con.SWP_NOSIZE)
            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
            img =  ImageGrab.grab(bbox = (left+5, top+180, right-10, bot-230))
            img_np = numpy.array(img)
            self._scene = img_np
            #二值化黑棋跟白棋的影像
            ret,self._white_scene = cv2.threshold(img_np,190,255,cv2.THRESH_BINARY)
            ret,self._black_scene = cv2.threshold(img_np,38,255,cv2.THRESH_BINARY_INV)
        except :
            print("找不到視窗")
            sys.exit()

        return True  # Success
#180----------------------------------------------------------------------------
#可刪除
    def _updateBoardHomography(self):
        if self._lastCorners is not None:
            corners = numpy.array(self._lastCorners, numpy.float32)
            corners = corners.reshape(self._numCorners, 2) #將其轉換成二維7*7的陣列 以符合我們的棋盤
            self._lastCorners = corners


    def _updateBoard(self):

        self._white_squareStatuses.fill(SQUARE_STATUS_UNKNOWN)  #每次重新偵測要清除 否則會留下上一個的
        self._black_squareStatuses.fill(SQUARE_STATUS_UNKNOWN)
        #白棋影像處理
        for i in range(self._patternSize[0] + 1): #生成每個方格顏色的資料
            for j in range(self._patternSize[1] + 1):
                self._updateSquareData("white",i, j)
        for i in range(self._patternSize[0] + 1): #更新每個方格的分類
            for j in range(self._patternSize[1] + 1):
                self._updateSquareStatus("white",i, j)
                
        #黑棋影像處理
        for i in range(self._patternSize[0] + 1): #生成每個方格顏色的資料
            for j in range(self._patternSize[1] + 1):
                self._updateSquareData("black",i, j)
        for i in range(self._patternSize[0] + 1): #更新每個方格的分類
            for j in range(self._patternSize[1] + 1):
                self._updateSquareStatus("black",i, j)

#185----------------------------------------------------------------------------
#計算每個方格的顏色的資料
    def _updateSquareData(self, chesscolor, i, j):

        self._squareFreqs[j, i] = None

        x0 = i * self._squareWidth #計算方格四角的座標
        x1 = x0 + self._squareWidth
        y0 = j * self._squareWidth
        y1 = y0 + self._squareWidth

        #K-means會自動將你的像素分為兩個主要顏色 並且把資料進行分類並存取
        #輸入必須是一個2D NUMPY陣列
        #根據P169 前面兩個是YX 故從y0擷取到y1 然後x0擷取到x1 然後reshape成(75*75,3)
        if chesscolor == "white":
            self._clusterer.fit(
                    self._white_scene[y0:y1, x0:x1].reshape(
                            self._squareArea, 3))
        elif chesscolor == "black":
            self._clusterer.fit(
                    self._black_scene[y0:y1, x0:x1].reshape(
                            self._squareArea, 3))

        #labels為一個一維陣列 其長度等於方格內素數量 如果一個像素與第一個主要顏色聚集在一起 為0 與第二個顏色聚集在一起為1 所以我們用mean平均可以來計算第二個顏色的"頻率"是多少
        freq = numpy.mean(self._clusterer.labels_)
        if freq > 0.5:
            freq = 1.0 - freq

        self._squareFreqs[j, i] = freq #儲存該格的頻率

#186~187------------------------------------------------------------------------

    def _updateSquareStatus(self, chesscolor, i, j):

        freq = self._squareFreqs[j, i] #首先得知方格的頻率

        if freq < self.emptyFreqThreshold: #1.小於空格門檻 代表是空格 分類為空格 squareStatus表示狀態 什麼都沒偵測到則=0 白棋=1 黑棋=2
            squareStatus = SQUARE_STATUS_EMPTY
        else:
                if chesscolor == "white" :
                    squareStatus = "1"
                    self._white_squareStatuses[j, i] = squareStatus #儲存該格的分類結果
                elif chesscolor == "black":
                    squareStatus = "2"
                    self._black_squareStatuses[j, i] = squareStatus #儲存該格的分類結果


#178,189------------------------------------------------------------------------
#繪製文字於方格上 update的最後一步
    def _drawSquareStatus(self, i, j, AI=None):

        x0 = i * self._squareWidth
        y0 = j * self._squareWidth

        white_squareStatus = self._white_squareStatuses[j, i]
        if white_squareStatus > 0:
            text = str(white_squareStatus)
            textSize, textBaseline = cv2.getTextSize(  #選擇HERSHEY_PLAIN字型
                    text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
            xCenter = x0 + self._squareWidth // 2 #因為要繪製於方格的中心
            yCenter = y0 + self._squareWidth // 2
            textCenter = (xCenter - textSize[0] // 2, #並且還要往左移一點才會看起來真的是中 否則只是從中心"開始"繪製
                          yCenter + textBaseline)
            cv2.putText(self._white_scene, text, textCenter, #放置文字
                        cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0), 1, cv2.LINE_AA) #cv2.LINE_AA反鋸齒
            cv2.putText(self._scene, text, textCenter, #放置文字
                        cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0), 1, cv2.LINE_AA) #cv2.LINE_AA反鋸齒
        black_squareStatus = self._black_squareStatuses[j, i]
        if black_squareStatus > 0:
            text = str(black_squareStatus)
            textSize, textBaseline = cv2.getTextSize(  #選擇HERSHEY_PLAIN字型
                    text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
            xCenter = x0 + self._squareWidth // 2 #因為要繪製於方格的中心
            yCenter = y0 + self._squareWidth // 2
            textCenter = (xCenter - textSize[0] // 2, #並且還要往左移一點才會看起來真的是中 否則只是從中心"開始"繪製
                          yCenter + textBaseline)
            cv2.putText(self._black_scene, text, textCenter, #放置文字
                        cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0), 1, cv2.LINE_AA) #cv2.LINE_AA反鋸齒
            cv2.putText(self._scene, text, textCenter, #放置文字
                        cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0), 1, cv2.LINE_AA) #cv2.LINE_AA反鋸齒
        if AI == True:
            text = str("Here")
            textSize, textBaseline = cv2.getTextSize(  #選擇HERSHEY_PLAIN字型
                    text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
            xCenter = x0 + self._squareWidth // 2 #因為要繪製於方格的中心
            yCenter = y0 + self._squareWidth // 2
            textCenter = (xCenter - textSize[0] // 2, #並且還要往左移一點才會看起來真的是中 否則只是從中心"開始"繪製
                          yCenter + textBaseline)
            cv2.putText(self._scene, text, textCenter, #放置文字
                        cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0), 1, cv2.LINE_AA) #cv2.LINE_AA反鋸齒
#--------------------------------------------------------------------------
    """
    在update()中呼叫_updateBoard偵測完後呼叫此函式將棋盤陣列送至AI以取得AI的動作
    """
    def scanningToAI(self):

        self.board.states = {}
        self.board.moved = []
        #1.根據辨識結果轉化成棋盤狀態
        for i in range(8):
            for j in range(8):
                #    self.board.current_player=
    #因為我們的棋盤的行從上到下是從7到0 這裡是0~7 故要把7-列 改回來
                if self._white_squareStatuses[i][j]==1:
                    #print("在第%s列%s行move%s找到白棋"%(i,j,move))
                    move=self.board.location_to_move((7-i,j)) #取得該(i、j)代表的動作(位子)
                    self.board.moved.append(move) #加入進已使用的動作
                    self.board.states[move]=1 #使用字典儲存表示棋盤狀態
                if self._black_squareStatuses[i][j]==2:
                    move=self.board.location_to_move((7-i,j))
                    self.board.moved.append(move)
                    self.board.states[move]=2
        #2.開始掃描取得所有可下的點
        self.board.scanning_for_availables()
        #3.送進MCTS開始模擬最後取的AI的動作
        self.AImove = self.mcts_player.get_action(self.board)
        #如果沒給出AImove 沒偵測到，呼叫reupdatescene重新偵測畫面直到有給出AImove
        while(self.AImove==None):
            print("重新偵測 有畫面但程式沒偵測到棋盤或棋盤已滿故AImove給出了None而導致錯誤")
            self.reupdatescene()
        #4.將AI的動作轉為其代表的行列位子
        print("AI選擇",self.AImove)
        w,h = self.board.move_to_location(self.AImove)
        #5.回傳
        return (w,h)

#跟update()函式原理差不多 
    def reupdatescene(self):
        self._updateScene() #1.讀取影像
        self._updateBoard() #2.辨識影像

        #3.scanningToAI()首先根據辨識結果轉化成棋盤狀態 再送進AI取得動作
        self.board.states = {}
        self.board.moved = []
        for i in range(8):
            for j in range(8):
                if self._white_squareStatuses[i][j]==1:
                    move=self.board.location_to_move((7-i,j)) #因為我們的棋盤的行從上到下是從7到0 這裡是0~7 故要把7-列 改回來
#                    print("在第%s列%s行move%s找到白棋"%(i,j,move))
                    self.board.moved.append(move)
                    self.board.states[move]=1
                if self._black_squareStatuses[i][j]==2:
                    move=self.board.location_to_move((7-i,j))

                    self.board.moved.append(move)
                    self.board.states[move]=2
        self.board.scanning_for_availables()

        self.AImove = self.mcts_player.get_action(self.board)













