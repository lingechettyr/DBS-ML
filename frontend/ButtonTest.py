'''
.py file with button code separated from the other code
'''




import sys
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QMainWindow, QFrame
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QPoint
from PyQt5 import uic, QtCore, QtGui, QtWidgets

import images

if hasattr(QtCore.Qt, 'AA_DisableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)

DownloadMenuOn = False
LoadingProcess = False

#Main App
class TheApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 700)
        
        uic.loadUi('ProgramWindow.ui', self)

    
        self.stackedWidget.setCurrentIndex(0)

        #BtnFunc
        self.Graph1Button.clicked.connect(self.change2)
        self.Graph2Button.clicked.connect(self.change2)
        self.Graph3Button.clicked.connect(self.change2)
        self.Graph4Button.clicked.connect(self.change2)

        self.QuitButton.clicked.connect(self.change1)
      
 
    def change1(self):
        self.stackedWidget.setCurrentIndex(0)
        print('Change1TestPassed')

    def change2(self):
        self.stackedWidget.setCurrentIndex(1)
        print('Change2TestPassed')

    

#init
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
  
    MyApp = TheApp()
    MyApp.show()

    sys.exit(app.exec_())
