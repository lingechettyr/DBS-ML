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

        self.currentgraph = 0
 
        self.stackedWidget.setCurrentIndex(0)

        #BtnFunc
        self.Graph1Button.clicked.connect(self.LoadingLaunch)
        self.Graph1Button.clicked.connect(self.gr1)
        self.Graph2Button.clicked.connect(self.LoadingLaunch)
        self.Graph2Button.clicked.connect(self.gr2)
        self.Graph3Button.clicked.connect(self.LoadingLaunch)
        self.Graph3Button.clicked.connect(self.gr3)
        self.Graph4Button.clicked.connect(self.LoadingLaunch)
        self.Graph4Button.clicked.connect(self.gr4)

        self.QuitButton.clicked.connect(self.change1)

        
        self.DownloadImage.clicked.connect(self.switchDownloadMenu)
        
    
        #LoadBarAnimation
    def LoadingLaunch(self):
        global LoadingProcess

        if LoadingProcess == False:
            self.animation() 
            LoadingProcess = True
            return LoadingProcess

    def LoadingStop(self):
        self.change2()
        global LoadingProcess
        LoadingProcess = False
        return LoadingProcess


    def animation(self):
        self.bar = QPropertyAnimation(self.LoadingBar, b'geometry')
        self.bar.setDuration(1500)
       
        self.bar.setKeyValueAt(0, QRect(self.LoadingBar.x(), self.LoadingBar.y(), 15, 15))
        self.bar.setKeyValueAt(0.1, QRect(self.LoadingBar.x(), self.LoadingBar.y() - 60, 15, 15))
        self.bar.setKeyValueAt(0.2, QRect(self.LoadingBar.x(), self.LoadingBar.y() - 40, 15, 15))
        self.bar.setKeyValueAt(0.9, QRect(self.LoadingBar.x() - 140, self.LoadingBar.y() - 40, 295, 15))
        self.bar.setKeyValueAt(0.95, QRect(self.LoadingBar.x(), self.LoadingBar.y() - 60, 15, 15))
        self.bar.setKeyValueAt(1, QRect(self.LoadingBar.x(), self.LoadingBar.y(), 15, 15))

        self.bar.start()
        self.bar.finished.connect(self.LoadingStop)
        
    def change1(self):
        self.stackedWidget.setCurrentIndex(0)
        self.currentgraph = 0

    def change2(self):
        self.stackedWidget.setCurrentIndex(1)
        if self.currentgraph == 1:
            self.Frame3_2.setStyleSheet(u"background-color: rgb(43, 43, 58);\n"
"image: url(:/DownloadImage/Graphs/Graph1.jpg);")
        elif self.currentgraph == 2:
            self.Frame3_2.setStyleSheet(u"background-color: rgb(43, 43, 58);\n"
"image: url(:/DownloadImage/Graphs/Graph2.jpg);")
        elif self.currentgraph == 3:
            self.Frame3_2.setStyleSheet(u"background-color: rgb(43, 43, 58);\n"
"image: url(:/DownloadImage/Graphs/Graph3.jpg);")
        elif self.currentgraph == 4:
            self.Frame3_2.setStyleSheet(u"background-color: rgb(43, 43, 58);\n"
"image: url(:/DownloadImage/Graphs/Graph4.jpg);")

    def gr1(self):
        self.currentgraph = 1
        
    def gr2(self):
        self.currentgraph = 2

    def gr3(self):
        self.currentgraph = 3

    def gr4(self):
        self.currentgraph = 4


    def switchDownloadMenu(self):
        global DownloadMenuOn

        self.pictd = QPropertyAnimation(self.JPEGdownload, b'geometry')
        self.pictd.setDuration(100)
       
        self.pictd.setKeyValueAt(0, QRect(self.JPEGdownload.x(), self.JPEGdownload.y(), 40, 40))
        self.pictd.setKeyValueAt(1, QRect(self.JPEGdownload.x() - 60, self.JPEGdownload.y(), 40, 40))
        
        self.pictd2 = QPropertyAnimation(self.PNGdownload, b'geometry')
        self.pictd2.setDuration(100)
      
        self.pictd2.setKeyValueAt(0, QRect(self.PNGdownload.x(), self.PNGdownload.y(), 40, 40))
        self.pictd2.setKeyValueAt(1, QRect(self.PNGdownload.x() - 60, self.PNGdownload.y(), 40, 40))

        self.pictdh = QPropertyAnimation(self.JPEGdownload, b'geometry')
        self.pictdh.setDuration(100)
    
        self.pictdh.setKeyValueAt(0, QRect(self.JPEGdownload.x(), self.JPEGdownload.y(), 40, 40))
        self.pictdh.setKeyValueAt(1, QRect(self.JPEGdownload.x() + 60, self.JPEGdownload.y(), 40, 40))
        
        self.pictdh2 = QPropertyAnimation(self.PNGdownload, b'geometry')
        self.pictdh2.setDuration(100)
   
        self.pictdh2.setKeyValueAt(0, QRect(self.PNGdownload.x(), self.PNGdownload.y(), 40, 40))
        self.pictdh2.setKeyValueAt(1, QRect(self.PNGdownload.x() + 60, self.PNGdownload.y(), 40, 40))

        if DownloadMenuOn == False:
            self.DownloadImage.setStyleSheet(u"image: url(:/DownloadImage/DownloadInverted.png);\n"
"background-color: rgba(250, 250, 250, 0);\n"
"border-color: rgba(250, 250, 250, 0)")
            self.pictd.start()
            self.pictd2.start()
            DownloadMenuOn = True
            return DownloadMenuOn

        elif DownloadMenuOn == True:
            self.DownloadImage.setStyleSheet(u"image: url(:/DownloadImage/Download.png);\n"
"background-color: rgba(250, 250, 250, 0);\n"
"border-color: rgba(250, 250, 250, 0)")
            self.pictdh.start()
            self.pictdh2.start()
            DownloadMenuOn = False
            return DownloadMenuOn



#init

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    #set widgets to show
    MyApp = TheApp()
    MyApp.show()

    sys.exit(app.exec_())
