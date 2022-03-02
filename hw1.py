import os
import sys
import glob
import scipy
import argparse
import cv2 as cv
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QMainWindow, QApplication,QTextEdit
from hw1_ui import Ui_MainWindow
from scipy import signal
from scipy import misc
import imutils

cont3 = 0
row = 250
column = 461
sobelxImage = np.zeros((row,column))
sobelyImage = np.zeros((row,column))
sobelGrad = np.zeros((row,column))
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
        
    def onBindingUI(self):
        
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn4_3.clicked.connect(self.on_btn4_3_click)
        self.btn4_4.clicked.connect(self.on_btn4_4_click)
        # self.btn5_1.clicked.connect(self.on_btn5_1_click)
        # self.btn5_2.clicked.connect(self.on_btn5_2_click)
        # self.btn5_3.clicked.connect(self.on_btn5_3_click)
        # self.btn5_4.clicked.connect(self.on_btn5_4_click)
        # self.btn5_5.clicked.connect(self.on_btn5_5_click)

#1
    def on_btn1_1_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        cv.imshow('Hw1-1.jpg', img)
        print('Height =', img.shape[0])
        print('Width =', img.shape[1])

    def on_btn1_2_click(self):
         img = cv.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
         (B, G, R) = cv.split(img)
         zeros = np.zeros(img.shape[:2], dtype='uint8') 
         b = cv.merge([B, zeros, zeros])  
         g = cv.merge([zeros, G, zeros])
         r = cv.merge([zeros, zeros, R])
         cv.imshow('Sun.jpg', img)
         cv.imshow('b.jpg', b)
         cv.imshow('g.jpg', g)
         cv.imshow('r.jpg', r)

    def on_btn1_3_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('Sun.jpg',img)
        cv.imshow('Gray scale using Opencv function', gray)

        [row, col, channel] = img.shape
        avg = np.zeros((row, col),np.uint8)
        for i in range(row):
            for j in range(col):
                (b, g, r) = img[i, j]
                gray = (int(b)+int(g)+int(r))/3
                avg[i,j] = np.uint8(gray)
        cv.imshow("Gray scale by average method", avg)

    def on_btn1_4_click(self):
        img1 = cv.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg')
        img2 = cv.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg')

        def blend(x):
            percentage = cv.getTrackbarPos('Blend', 'Blending')
            alpha = 1 - percentage / 255
            beta = percentage / 255
            img_blended = cv.addWeighted(img1, alpha, img2, beta, 0)
            cv.imshow('Blending', img_blended)

        cv.namedWindow('Blending')
        cv.createTrackbar('Blend', 'Blending', 0, 255, blend)
        cv.imshow('Blending', img1)

#2
    def on_btn2_1_click(self):
         img = cv.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')  
         cv.imshow('Original',img)
         image_gaussian_processed = cv.GaussianBlur(img,(5,5),1)        
         cv.imshow('Gaussian blur',image_gaussian_processed)

    def on_btn2_2_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')
        Bilateral = cv.bilateralFilter(img ,9,90,90)        
        cv.imshow('Original Image',img)
        cv.imshow('bilateralFilter',Bilateral)

    def on_btn2_3_click(self):
         img = cv.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg')
         cv.imshow('Original',img)
         median = cv.medianBlur(img,3)  
         cv.imshow('3x3',median)
         median1 = cv.medianBlur(img,5)  
         cv.imshow('5x5',median1)           
        
#3
    def on_btn3_1_click(self): 
        House = cv.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')       
        cv.imshow('House',House)

        grayHouse = cv.cvtColor(House,cv.COLOR_RGB2GRAY)
        cv.imshow('Grayscale',grayHouse)

        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        [row, col] = grayHouse.shape
        gray_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gray_extand[i+1][j+1] = grayHouse[i][j]
        Gaussian_Blur = np.zeros((row, col),np.uint8)
        #convolution
        for i in range(row):
            for j in range(col):
                Gaussian_Blur [i][j] = (gray_extand[i][j]*gaussian_kernel[0][0] + gray_extand[i][j+1]*gaussian_kernel[0][1] + gray_extand[i][j+2]*gaussian_kernel[0][2]
                + gray_extand[i+1][j]*gaussian_kernel[1][0] + gray_extand[i+1][j+1]*gaussian_kernel[1][1] + gray_extand[i+1][j+2]*gaussian_kernel[1][2]
                + gray_extand[i+2][j]*gaussian_kernel[2][0] + gray_extand[i+2][j+1]*gaussian_kernel[2][1] + gray_extand[i+2][j+2]*gaussian_kernel[2][2])
        cv.imshow('Gaussian Blur',Gaussian_Blur)

    def on_btn3_2_click(self): 
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        [row, col] = gray.shape
        gray_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gray_extand[i+1][j+1] = gray[i][j]
        Gaussian_Blur = np.zeros((row, col),np.uint8)
        #convolution
        for i in range(row):
            for j in range(col):
                Gaussian_Blur[i][j] = (gray_extand[i][j]*gaussian_kernel[0][0] + gray_extand[i][j+1]*gaussian_kernel[0][1] + gray_extand[i][j+2]*gaussian_kernel[0][2]
                + gray_extand[i+1][j]*gaussian_kernel[1][0] + gray_extand[i+1][j+1]*gaussian_kernel[1][1] + gray_extand[i+1][j+2]*gaussian_kernel[1][2]
                + gray_extand[i+2][j]*gaussian_kernel[2][0] + gray_extand[i+2][j+1]*gaussian_kernel[2][1] + gray_extand[i+2][j+2]*gaussian_kernel[2][2])

        # Sobel X
        s_x = np.array([
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
                ])
        # extand
        gau_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gau_extand[i+1][j+1] = Gaussian_Blur[i][j]
        #convolution
        sobel_x = np.zeros((row, col),np.uint8)
        for i in range(row):
            for j in range(col):
                sobel_x[i][j] = abs(gau_extand[i][j]*s_x[0][0] + gau_extand[i][j+1]*s_x[0][1] + gau_extand[i][j+2]*s_x[0][2]
                + gau_extand[i+1][j]*s_x[1][0] + gau_extand[i+1][j+1]*s_x[1][1] + gau_extand[i+1][j+2]*s_x[1][2]
                + gau_extand[i+2][j]*s_x[2][0] + gau_extand[i+2][j+1]*s_x[2][1] + gau_extand[i+2][j+2]*s_x[2][2])
        cv.imshow('sobel X',sobel_x)
    def on_btn3_3_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        [row, col] = gray.shape
        gray_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gray_extand[i+1][j+1] = gray[i][j]
        Gaussian_Blur = np.zeros((row, col),np.uint8)
        #convolution
        for i in range(row):
            for j in range(col):
                Gaussian_Blur[i][j] = (gray_extand[i][j]*gaussian_kernel[0][0] + gray_extand[i][j+1]*gaussian_kernel[0][1] + gray_extand[i][j+2]*gaussian_kernel[0][2]
                + gray_extand[i+1][j]*gaussian_kernel[1][0] + gray_extand[i+1][j+1]*gaussian_kernel[1][1] + gray_extand[i+1][j+2]*gaussian_kernel[1][2]
                + gray_extand[i+2][j]*gaussian_kernel[2][0] + gray_extand[i+2][j+1]*gaussian_kernel[2][1] + gray_extand[i+2][j+2]*gaussian_kernel[2][2])

        # Sobel Y
        s_y = np.array([
                [1,2,1],
                [0,0,0],
                [-1,-2,-1]
                ])
        sobel_y = np.zeros((row-4, col-4),np.uint8)
        #convolution
        for i in range(row-4):
            for j in range(col-4):
                sobel_y[i][j] = abs(Gaussian_Blur[i][j]*s_y[0][0] + Gaussian_Blur[i][j+1]*s_y[0][1] + Gaussian_Blur[i][j+2]*s_y[0][2]
                + Gaussian_Blur[i+1][j]*s_y[1][0] + Gaussian_Blur[i+1][j+1]*s_y[1][1] + Gaussian_Blur[i+1][j+2]*s_y[1][2]
                + Gaussian_Blur[i+2][j]*s_y[2][0] + Gaussian_Blur[i+2][j+1]*s_y[2][1] + Gaussian_Blur[i+2][j+2]*s_y[2][2])

        cv.imshow('sobel y',sobel_y)            

    def on_btn3_4_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        [row, col] = gray.shape
        gray_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gray_extand[i+1][j+1] = gray[i][j]
        img_gau = np.zeros((row, col),np.uint8)
        #convolution
        for i in range(row):
            for j in range(col):
                img_gau[i][j] = (gray_extand[i][j]*gaussian_kernel[0][0] + gray_extand[i][j+1]*gaussian_kernel[0][1] + gray_extand[i][j+2]*gaussian_kernel[0][2]
                + gray_extand[i+1][j]*gaussian_kernel[1][0] + gray_extand[i+1][j+1]*gaussian_kernel[1][1] + gray_extand[i+1][j+2]*gaussian_kernel[1][2]
                + gray_extand[i+2][j]*gaussian_kernel[2][0] + gray_extand[i+2][j+1]*gaussian_kernel[2][1] + gray_extand[i+2][j+2]*gaussian_kernel[2][2])

        # Sobel X
        s_x = np.array([
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
                ])
        # extand
        gau_extand = np.zeros((row+2, col+2),np.uint8)
        for i in range(row):
            for j in range(col):
                gau_extand[i+1][j+1] = img_gau[i][j]
        #convolution
        sobel_x = np.zeros((row, col),np.uint8)
        for i in range(row):
            for j in range(col):
                sobel_x[i][j] = abs(gau_extand[i][j]*s_x[0][0] + gau_extand[i][j+1]*s_x[0][1] + gau_extand[i][j+2]*s_x[0][2]
                + gau_extand[i+1][j]*s_x[1][0] + gau_extand[i+1][j+1]*s_x[1][1] + gau_extand[i+1][j+2]*s_x[1][2]
                + gau_extand[i+2][j]*s_x[2][0] + gau_extand[i+2][j+1]*s_x[2][1] + gau_extand[i+2][j+2]*s_x[2][2])
        # Sobel Y
        s_y = np.array([
                [1,2,1],
                [0,0,0],
                [-1,-2,-1]
                ])
        sobel_y = np.zeros((row-4, col-4),np.uint8)
        #convolution
        for i in range(row-4):
            for j in range(col-4):
                sobel_y[i][j] = abs(img_gau[i][j]*s_y[0][0] + img_gau[i][j+1]*s_y[0][1] + img_gau[i][j+2]*s_y[0][2]
                + img_gau[i+1][j]*s_y[1][0] + img_gau[i+1][j+1]*s_y[1][1] + img_gau[i+1][j+2]*s_y[1][2]
                + img_gau[i+2][j]*s_y[2][0] + img_gau[i+2][j+1]*s_y[2][1] + img_gau[i+2][j+2]*s_y[2][2])

        #magnitude
        mag = np.zeros((row-4, col-4),np.uint8)
        for i in range(row-4):
            for j in range(col-4):
                mag[i][j] = (sobel_x[i][j]**2+sobel_y[i][j]**2)**0.5

        cv.imshow('magnutude',mag)

#4    
    def on_btn4_1_click(self):       
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        # resize
        resize=self.text1.toPlainText()
        img_resize = cv.resize(img, (int(resize),int(resize)))
        cv.imshow("resize", img_resize)

    def on_btn4_2_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        resize=self.text1.toPlainText()
        Tx=self.text2.toPlainText()
        Ty=self.text3.toPlainText()
        img_resize = cv.resize(img, (int(resize),int(resize)))
        # translate
        T = np.float32([[1, 0, int(Tx)], [0, 1, int(Ty)]])
        img_tran = cv.warpAffine(img_resize, T, (400,300))
        cv.imshow("translate", img_tran)

    def on_btn4_3_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        resize=self.text1.toPlainText()
        Tx=self.text2.toPlainText()
        Ty=self.text3.toPlainText()
        img_resize = cv.resize(img, (int(resize),int(resize)))
        T = np.float32([[1, 0, int(Tx)], [0, 1, int(Ty)]])
        img_tran = cv.warpAffine(img_resize, T, (400,300))
        
        # rotate
        (h, w) = img_resize.shape[:2]
        (cx, cy) = (w//2 ,h//2)
        rotate=self.text4.toPlainText()
        R = cv.getRotationMatrix2D((cx, cy), int(rotate), 1.0)
        img_rotate = cv.warpAffine(img_resize, R, (400, 300))
        img_rotate = imutils.rotate_bound(img_resize, -int(rotate))
        #cv.imshow('rotate',img_rotate)

        # scale
        scale = self.text5.toPlainText()
        img_scale = cv.resize(img_rotate, None, fx=float(scale), fy=float(scale))
        t = np.float32([[1, 0, img_resize.shape[1] / 4], [0, 1, 60]])
        img_scale = cv.warpAffine(img_scale, t, (400, 300))
        cv.imshow("rotate and scale", img_scale)
        
    def on_btn4_4_click(self):
        img = cv.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        resize=self.text1.toPlainText()
        Tx=self.text2.toPlainText()
        Ty=self.text3.toPlainText()
        img_resize = cv.resize(img, (int(resize),int(resize)))
        T = np.float32([[1, 0, Tx], [0, 1, Ty]])
        img_tran = cv.warpAffine(img_resize, T, (400,300))
        
        # rotate
        (h, w) = img_resize.shape[:2]
        (cx, cy) = (w//2 ,h//2)
        rotate=self.text4.toPlainText()
        R = cv.getRotationMatrix2D((cx, cy), int(rotate), 1.0)
        img_rotate = cv.warpAffine(img_resize, R, (400, 300))
        img_rotate = imutils.rotate_bound(img_resize, -int(rotate))
        #cv.imshow('rotate',img_rotate)

        # scale
        scale = self.text5.toPlainText()
        img_scale = cv.resize(img_rotate, None, fx=float(scale), fy=float(scale))
        t = np.float32([[1, 0, img_resize.shape[1] / 4], [0, 1, 60]])
        img_scale = cv.warpAffine(img_scale, t, (400, 300))
        #cv.imshow("rotate and scale", img_scale)

        # shear
        (row, col, ch) = img.shape
        shearx1=self.text6_1.toPlainText()
        shearx2=self.text6_2.toPlainText()
        sheary1=self.text6_3.toPlainText()
        sheary2=self.text6_4.toPlainText()
        shearz1=self.text6_5.toPlainText()
        shearz2=self.text6_6.toPlainText()
        pts1 = np.float32([[50,50], [200,50], [50,200]])
        pts2 = np.float32([[int(shearx1),int(shearx2)], [int(sheary1),int(sheary2)], [int(shearz1),int(shearz2)]])
        S = cv.getAffineTransform(pts1, pts2)
        img_shear = cv.warpAffine(img_scale, S, (400,300))
        cv.imshow("shearing", img_shear)
    
#5

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())