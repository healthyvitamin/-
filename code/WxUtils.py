
import wx

#189~191---------------------------------------------------------------------------
# Try to determine whether we are on Raspberry Pi. 檢查是不是樹梅派電腦 因為會有BUG在轉圖片上 必須多一個步驟轉圖片
IS_RASPBERRY_PI = False
try:
    with open('/proc/cpuinfo') as f: #見課本
        for line in f:
            line = line.strip()
            if line.startswith('Hardware') and \
                    line.endswith('BCM2708'):
                IS_RASPBERRY_PI = True
                break
except:
    pass

if IS_RASPBERRY_PI:
    def wxBitmapFromCvImage(image):
        h, w = image.shape[:2]
        wxImage = wx.ImageFromBuffer(w, h, image)
        bitmap = wx.Bitmap.FromImage(wxImage)
        return bitmap
else:
    def wxBitmapFromCvImage(image):
        h, w = image.shape[:2]
        # The following conversion fails on Raspberry Pi.
        bitmap = wx.Bitmap.FromBuffer(w, h, image)
        return bitmap
