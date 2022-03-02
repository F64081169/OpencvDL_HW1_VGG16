import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os
import cv2 as cv
from keras.models import Sequential
from keras.models import load_model
import sys
from q5_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication,QTextEdit
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        self.batch=32
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1.clicked.connect(self.on_btn1_click)
        self.btn2.clicked.connect(self.on_btn2_click)
        self.btn3.clicked.connect(self.on_btn3_click)
        self.btn4.clicked.connect(self.on_btn4_click)
        self.btn5.clicked.connect(self.on_btn5_click)

    def on_btn1_click(self):
        index=0
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(17, 8))
        for i in range(3):
            for j in range(3):
                axes[i,j].set_title(labels[y_train[index][0]])
                
                axes[i,j].imshow(x_train[index])
                axes[i,j].get_xaxis().set_visible(False)
                axes[i,j].get_yaxis().set_visible(False)
                index += 1
        
        plt.show()

    def on_btn2_click(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        config = optimizer.get_config()
        print("hyperparameters:")
        print("batch size: ",self.batch)
        print("learning rate: ", config["learning_rate"])
        print("optimizer: ", config["name"])

    def on_btn3_click(self):
        model = load_model('model.h5')
        model.summary()
        
    def on_btn4_click(self):
        img = cv.imread('train.py_loss_and_accuracy_plot.png')
        cv.imshow('accuracy',img)

    def on_btn5_click(self):
        num=int(self.textEdit.toPlainText())
        model = load_model('model.h5')
        x_train_normalize = x_train.astype('float32') / 255.0
        x_test_normalize = x_test.astype('float32') / 255.0
        prediction = model.predict(x_test_normalize)
        print(prediction)
        Arr = []
        for j in range(10):
            Arr.append('%1.9f'% (prediction[num][j]))
        pred_Arr = list(map(float,Arr))   
        y=np.arange(0, 1, 0.1)
        plt.figure(figsize=(10,5)) 
        plt.yticks(y)   
        plt.bar(labels, pred_Arr)

        plt.figure(figsize=(5,5))
        plt.imshow(np.reshape(x_test[num],(32, 32,3)))
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())