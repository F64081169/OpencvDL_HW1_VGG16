import sys
from PyQt5.QtWidgets import QMainWindow, QApplication,QTextEdit
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets,uic

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
          super(Ui_MainWindow, self).__init__() # Call the inherited classes _init_ method
          uic.loadUi('untitled.ui', self) # Load the .ui file
          self.show() # Show the GUI