# -*- coding: utf-8 -*-

import threading
import wx

import CheckersModel
import WxUtils
#192-------------------------------------------------------------------------
class Checkers(wx.Frame):


    def __init__(self, checkersModel, title='Checkers'): #用wx套件開始建立GUI
        self._checkersModel = checkersModel
        #詳見網路 wx視窗的各種style設定 這邊依序是關閉按鈕、最小化按鈕、標題欄、系統選單、可變大小
        style = wx.CLOSE_BOX | wx.MINIMIZE_BOX | wx.CAPTION | \
                wx.SYSTEM_MENU | wx.CLIP_CHILDREN
        wx.Frame.__init__(self, None, title=title, style=style)
        self.SetBackgroundColour(wx.Colour(232, 232, 232)) #背景為灰色

        self.Bind(wx.EVT_CLOSE, self._onCloseWindow) #視窗關閉時呼叫closewindow fc做手動清除
        #1.給事件綁定識別器 前面是視窗關閉都會做的事 這裡是按下esc時的處理
        quitCommandID = wx.NewId()
        self.Bind(wx.EVT_MENU, self._onQuitCommand, #將"關閉fc"綁定到wx.EVT_MENU(一個事件處理器) 並且設定一個id
                  id=quitCommandID)
        #2.將綁定連結到鍵盤ESC鍵上
        acceleratorTable = wx.AcceleratorTable([ #把ESC這個按鈕綁定到id上就完成了esc關閉功能 wx.ACCEL_NORMAL代表沒有別的model
            (wx.ACCEL_NORMAL, wx.WXK_ESCAPE,
             quitCommandID)
        ])
        self.SetAcceleratorTable(acceleratorTable)
#193-------------------------------------------------------------------------
#_sceneStaticBitmap用於顯示現實棋盤(全彩)
#_whiteStaticBitmap、_blackStaticBitmap則是二值化後的棋盤 為了偵測棋子用
        self._sceneStaticBitmap = wx.StaticBitmap(self) #wx.StaticBitmap用於顯示"靜態"圖片的"容器"
        self._whiteStaticBitmap = wx.StaticBitmap(self)
        self._blackStaticBitmap = wx.StaticBitmap(self)
        self._showImages() #先顯示三個影像 隨後在副執行緒執行的_runCaptureLoop 更新影像就是呼叫它
        videosSizer = wx.BoxSizer(wx.HORIZONTAL) #BoxSizer 佈置工具時要加入的集合 這邊我們把bitmap轉成水平佈置
        videosSizer.Add(self._sceneStaticBitmap)
        videosSizer.Add(self._whiteStaticBitmap)
        videosSizer.Add(self._blackStaticBitmap)
#195、159-------------------------------------------------------------------------
#建立一個滑桿用於辨識圖片上棋子是黑棋還是白棋用的門檻值
#另一個則是選擇AI要扮演的是白棋還是黑棋 以正確偵測棋盤以及MCTS搜尋

        emptyFreqThresholdSlider = self._createLabeledSlider( #使用我們自己寫的_createLabeledSlider函式建造滑桿 P195
                'Empty threshold',
                self._checkersModel.emptyFreqThreshold * 100,
                self._onEmptyFreqThresholdSelected)
#計算距離的滑桿 原是課本用於偵測顏色"對比"的以判斷西洋棋是深色、淺色棋子、陰影 但黑白棋不需要
#        playerDistThresholdSlider = self._createLabeledSlider(
#                'Player threshold',
#                self._checkersModel.playerDistThreshold * 100,
#                self._onPlayerDistThresholdSelected)

        colorList = ['white', 'black']

        RadioBoxSizer , self.RadioBox =self._createRadioBox(
                'white',
                self._onPlayerchange,
                colorList
                )
        self.RadioBox.SetSelection(0)



#196-------------------------------------------------------------------------
        controlsStyle = wx.ALIGN_CENTER_VERTICAL | wx.RIGHT 
        controlsBorder = 8

        controlsSizer = wx.BoxSizer(wx.HORIZONTAL)  
#跟場景棋盤一樣 我們將BUTTON使用水平佈置、並且垂直置中、8個像素填充在右手邊中
#詳見https://wxpython.org/Phoenix/docs/html/wx.Sizer.html#wx.Sizer.Add 第二個參數0是proportion 比例
        controlsSizer.Add(emptyFreqThresholdSlider, 0,
                          controlsStyle, controlsBorder)
#        controlsSizer.Add(playerDistThresholdSlider, 0,
#                          controlsStyle, controlsBorder)
        controlsSizer.Add(RadioBoxSizer, 0,
                          controlsStyle, controlsBorder)


        rootSizer = wx.BoxSizer(wx.VERTICAL) #巢狀化 我們要先顯示影像再顯示按鈕 所以分次videosSizer以及controlsSizer
        rootSizer.Add(videosSizer)
        rootSizer.Add(controlsSizer, 0, wx.EXPAND | wx.ALL,
                      border=controlsBorder)
        self.SetSizerAndFit(rootSizer) #wx的SetSizerAndFit方法 可以讓我們的視窗修改自己的尺寸來符合sizer佈置
#197-------------------------------------------------------------------------
        self._captureThread = threading.Thread( #由於我們更新影像如果在主程式運行會永遠卡在這 所以要用thread另外運行於背景
            target=self._runCaptureLoop)
        self._running = True #用以控制更新迴圈的結束
        self._captureThread.start() #開始運行_runCaptureLoop
#195-------------------------------------------------------------------------
    def _createLabeledSlider(self, label, initialValue,
                             callback):

        slider = wx.Slider(self) #wx內建的滑桿
        slider.SetValue(initialValue) #所以比如emptyFreqThreshold 他的"初始值"就是0.3*100=30 而範圍是1~100
        slider.Bind(wx.EVT_SLIDER, callback) #將slider事件綁定至callback fc fc寫在P199 _onEmptyFreqThresholdSelected函式

        staticText = wx.StaticText(self, label=label) #滑桿名字

        sizer = wx.BoxSizer(wx.VERTICAL) #建立sizer設定垂直置中
        sizer.Add(slider, 0, wx.ALIGN_CENTER_HORIZONTAL)
        sizer.Add(staticText, 0, wx.ALIGN_CENTER_HORIZONTAL)
        return sizer #回傳sizer 
    
    def _createRadioBox(self, label, callback, color):

        RadioBox = wx.RadioBox(self,choices = color) #wx內建的滑桿
      #  RadioButton.SetValue(initialValue) #所以比如emptyFreqThreshold 他的"初始值"就是0.3*100=30 而範圍是1~100
        RadioBox.Bind(wx.EVT_RADIOBOX, callback) #將slider事件綁定至callback fc寫在P199 _onPlayerchange函式

        sizer = wx.BoxSizer(wx.VERTICAL) #建立sizer設定垂直置中
        sizer.Add(RadioBox, 0, wx.ALIGN_CENTER_HORIZONTAL)
        return sizer,RadioBox #回傳sizer跟radiobox sizer用於布置 radiobox回傳是為了這樣才能呼叫他的方法
#197-------------------------------------------------------------------------
    def _onCloseWindow(self, event):
        self._running = False
        self._captureThread.join() #join 等待_captureThread完成才繼續執行
        self.Destroy()
#198-------------------------------------------------------------------------
    def _onQuitCommand(self, event):
        self.Close()
#199------------------------------------------------------------------------

    def _onEmptyFreqThresholdSelected(self, event):  #因為我們實際要的是0.0~1.0的數值 故要*0.01
        self._checkersModel.emptyFreqThreshold = \
                event.Selection * 0.01

    def _onPlayerDistThresholdSelected(self, event):
        self._checkersModel.playerDistThreshold = \
                event.Selection * 0.01

    #切換棋子用，當選擇玩家的RadioBox變動時觸發
    def _onPlayerchange(self, event):
        #1.首先設定為False 讓一直運行的_captureThread關閉
        self._running = False
        print("切換棋子中")
        self._captureThread.join()
        #2.將目前玩家設定為RadioBox設定的值
        self._checkersModel.board.current_player=self._checkersModel.board.players[self.RadioBox.GetSelection()]
        #3.重新開始運行讓一直運行的_captureThread關閉進行偵測
        self._running = True
        print(self._captureThread.isAlive())
        self._captureThread = threading.Thread(target=self._runCaptureLoop)
        self._captureThread.start()
        print("目前運行的數量",threading.enumerate())

    def _runCaptureLoop(self):
        while self._running:
            if self._checkersModel.update(): #如果更新回傳成功 代表偵測到了影像
                wx.CallAfter(self._showImages) #用CallAfter呼叫"更新影像的方法"在"主執行緒"運行 所以_showImages fc不是在我們創造的thread中執行 而是運行GUI的主執行緒
#身為兩個執行緒無法同時存取GUI物件
#200------------------------------------------------------------------------

    def _showImages(self):
        #呼叫_showImage 傳入更新影像圖片 _checkersModel.scene是裝偵測到的圖片 _sceneStaticBitmap則是顯示的容器 sceneSize是用於當沒有傳入影像(沒偵測到時)要顯示的"空白"影像大小
        self._showImage(
            self._checkersModel.scene, self._sceneStaticBitmap,
            self._checkersModel.sceneSize) #見CheckersModel.py P175、176 一個是場景更新 一個是場景俯視圖
        self._showImage(
            self._checkersModel.white_scene, self._whiteStaticBitmap,
            self._checkersModel.sceneSize)
        self._showImage(
            self._checkersModel.black_scene, self._blackStaticBitmap,
            self._checkersModel.sceneSize)


    def _showImage(self, image, staticBitmap, size):
        if image is None:
            bitmap = wx.EmptyBitmap(size[0], size[1])
        else:
            bitmap = WxUtils.wxBitmapFromCvImage(image)#呼叫wxutils.py的fc來把cv圖轉成bitmap (為一個物件 不是陣列 print只會出現記憶體位子) (另外，CV圖是BGR格式不是RGB格式)
        staticBitmap.SetBitmap(bitmap) #顯示圖片

def main():
    checkersModel = CheckersModel.CheckersModel()
    app = wx.App()
    checkers = Checkers(checkersModel)
    checkers.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
