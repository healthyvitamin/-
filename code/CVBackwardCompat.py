import cv2


CV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])#取得目前版本
if CV_MAJOR_VERSION < 3:
    #解決opencv舊版本與新版本之間的參數名稱問題  詳見課本P167
    #當opencv版本小於3時，將3版本的新參數的名稱插入至此
    cv2.LINE_AA = cv2.CV_AA
    cv2.CAP_PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    cv2.CAP_PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
    cv2.FILLED = cv2.cv.CV_FILLED
