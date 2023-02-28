#encoding:utf-8
#BY：Eastmount CSDN 2018-08-06
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

#讀取圖片
imagePath = '123.jpg'
img = cv2.imread(imagePath)

#opencv預設的imread是以BGR的方式進行儲存的
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#灰度影象處理
GrayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(u"讀入lenna圖的shape為", GrayImage.shape)

#直方圖均衡化
#equ = cv2.equalizeHist(gray)

#高斯平滑 去噪
Gaussian = cv2.GaussianBlur(GrayImage, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
#Gaussian = cv2.GaussianBlur(GrayImage, (9, 9),0)

#中值濾波
Median = cv2.medianBlur(Gaussian, 5)

#Sobel運算元 XY方向求梯度 cv2.CV_8U
x = cv2.Sobel(Median, cv2.CV_32F, 1, 0, ksize = 3) #X方向
y = cv2.Sobel(Median, cv2.CV_32F, 0, 1, ksize = 3) #Y方向
#absX = cv2.convertScaleAbs(x)   # 轉回uint8    
#absY = cv2.convertScaleAbs(y)
#Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
gradient = cv2.subtract(x, y)
Sobel = cv2.convertScaleAbs(gradient)
cv2.imshow('dilation2', Sobel)
cv2.waitKey(0)

#二值化處理 周圍畫素影響
blurred = cv2.GaussianBlur(Sobel, (9, 9),0) #再進行一次高斯去噪
#注意170可以替換的
ret, Binary = cv2.threshold(blurred , 170, 255, cv2.THRESH_BINARY)
#cv2.imshow('dilation2', Binary)
cv2.waitKey(0)

# 膨脹和腐蝕操作的核函式
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
# 膨脹一次，讓輪廓突出
Dilation = cv2.dilate(Binary, element2, iterations = 1)
# 腐蝕一次，去掉細節
Erosion = cv2.erode(Dilation, element1, iterations = 1)
# 再次膨脹，讓輪廓明顯一些
Dilation2 = cv2.dilate(Erosion, element2,iterations = 3)
#cv2.imshow('Dilation2 ', Dilation2)
cv2.waitKey(0)


##########################################

#建立一個橢圓核函式
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
#執行影象形態學, 細節直接查文件，很簡單
closed = cv2.morphologyEx(Binary, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
#cv2.imshow('erode dilate', closed)
cv2.waitKey(0)

##########################################


#顯示圖形
titles = ['Source Image','Gray Image', 'Gaussian Image', 'Median Image',
          'Sobel Image', 'Binary Image', 'Dilation Image', 'Erosion Image', 'Dilation2 Image']  
images = [lenna_img, GrayImage, Gaussian,
          Median, Sobel, Binary,
          Dilation, Erosion, closed]  
for i in xrange(9):  
   plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')  
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()  

#cv2.imshow('Gray', GrayImage)
cv2.waitKey(0)

"""
接下來使用Dilation2圖片確定車牌的輪廓
這裡opencv3返回的是三個引數
  引數一：二值化影象
  引數二：輪廓型別 檢測的輪廓不建立等級關係
  引數三：處理近似方法  例如一個矩形輪廓只需4個點來儲存輪廓資訊
"""
(_, cnts, _) = cv2.findContours(closed.copy(), 
                                cv2.RETR_LIST,               #RETR_TREE
                                cv2.CHAIN_APPROX_SIMPLE)

#畫出輪廓
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
print(c)

#compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
print ('rectt', rect)
Box = np.int0(cv2.boxPoints(rect))
print ('Box', Box)

#draw a bounding box arounded the detected barcode and display the image
Final_img = cv2.drawContours(img.copy(), [Box], -1, (0, 0, 255), 3)

#cv2.imshow('Final_img', Final_img)
cv2.waitKey(0)