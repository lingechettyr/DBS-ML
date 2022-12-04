# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ProgramWindowfmphvP.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *

import images_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 700)
        MainWindow.setStyleSheet(u"def change1(self, class_instance):\n"
"	class_instance.setupUI(self.page_2)")
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(0, -30, 800, 700))
        self.stackedWidget.setStyleSheet(u"background-color: #22222e")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.Header1 = QFrame(self.page)
        self.Header1.setObjectName(u"Header1")
        self.Header1.setGeometry(QRect(-30, 30, 911, 91))
        self.Header1.setStyleSheet(u"background-color: #f66867")
        self.Header1.setFrameShape(QFrame.StyledPanel)
        self.Header1.setFrameShadow(QFrame.Raised)
        self.MenuText = QLabel(self.Header1)
        self.MenuText.setObjectName(u"MenuText")
        self.MenuText.setGeometry(QRect(380, 20, 101, 51))
        font = QFont()
        font.setFamily(u"Myriad Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.MenuText.setFont(font)
        self.MenuText.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.Graph4Button = QPushButton(self.page)
        self.Graph4Button.setObjectName(u"Graph4Button")
        self.Graph4Button.setGeometry(QRect(500, 450, 141, 131))
        self.Graph4Button.setStyleSheet(u"QPushButton {\n"
"background-color: #ffffff;\n"
"border: 2px solid #f66867;\n"
"	image: url(:/DownloadImage/Graphs/Graph4.jpg);\n"
"}\n"
"\n"
"")
        self.Graph3Frame = QFrame(self.page)
        self.Graph3Frame.setObjectName(u"Graph3Frame")
        self.Graph3Frame.setGeometry(QRect(160, 420, 141, 51))
        self.Graph3Frame.setStyleSheet(u"background-color:#f66867;\n"
"border-radius: 15")
        self.Graph3Frame.setFrameShape(QFrame.StyledPanel)
        self.Graph3Frame.setFrameShadow(QFrame.Raised)
        self.Graph3Text = QLabel(self.Graph3Frame)
        self.Graph3Text.setObjectName(u"Graph3Text")
        self.Graph3Text.setGeometry(QRect(10, -10, 71, 51))
        font1 = QFont()
        font1.setFamily(u"Myriad Pro")
        font1.setPointSize(9)
        font1.setBold(True)
        font1.setWeight(75)
        self.Graph3Text.setFont(font1)
        self.Graph3Text.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgba(255,255,255,0)")
        self.Graph1Frame = QFrame(self.page)
        self.Graph1Frame.setObjectName(u"Graph1Frame")
        self.Graph1Frame.setGeometry(QRect(160, 180, 141, 51))
        self.Graph1Frame.setStyleSheet(u"background-color: #f66867;\n"
"border-radius: 15")
        self.Graph1Frame.setFrameShape(QFrame.StyledPanel)
        self.Graph1Frame.setFrameShadow(QFrame.Raised)
        self.Graph1Text = QLabel(self.Graph1Frame)
        self.Graph1Text.setObjectName(u"Graph1Text")
        self.Graph1Text.setGeometry(QRect(10, -10, 71, 51))
        self.Graph1Text.setFont(font1)
        self.Graph1Text.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgba(255,255,255,0)")
        self.Graph2Frame = QFrame(self.page)
        self.Graph2Frame.setObjectName(u"Graph2Frame")
        self.Graph2Frame.setGeometry(QRect(500, 180, 141, 51))
        self.Graph2Frame.setStyleSheet(u"background-color: #f66867;\n"
"border-radius: 15")
        self.Graph2Frame.setFrameShape(QFrame.StyledPanel)
        self.Graph2Frame.setFrameShadow(QFrame.Raised)
        self.Graph2Text = QLabel(self.Graph2Frame)
        self.Graph2Text.setObjectName(u"Graph2Text")
        self.Graph2Text.setGeometry(QRect(10, -10, 71, 51))
        self.Graph2Text.setFont(font1)
        self.Graph2Text.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgba(255,255,255,0)")
        self.Graph1Button = QPushButton(self.page)
        self.Graph1Button.setObjectName(u"Graph1Button")
        self.Graph1Button.setGeometry(QRect(160, 210, 141, 131))
        self.Graph1Button.setStyleSheet(u"QPushButton {\n"
"background-color: #ffffff;\n"
"border: 2px solid #f66867;\n"
"	image: url(:/DownloadImage/Graphs/Graph1.jpg);\n"
"}\n"
"\n"
"")
        self.Graph2Button = QPushButton(self.page)
        self.Graph2Button.setObjectName(u"Graph2Button")
        self.Graph2Button.setGeometry(QRect(500, 210, 141, 131))
        self.Graph2Button.setStyleSheet(u"QPushButton {\n"
"background-color: #ffffff;\n"
"border: 2px solid #f66867;\n"
"	image: url(:/DownloadImage/Graphs/Graph2.jpg);\n"
"}\n"
"\n"
"")
        self.Graph4Frame = QFrame(self.page)
        self.Graph4Frame.setObjectName(u"Graph4Frame")
        self.Graph4Frame.setGeometry(QRect(500, 420, 141, 51))
        self.Graph4Frame.setStyleSheet(u"background-color: #f66867;\n"
"border-radius: 15")
        self.Graph4Frame.setFrameShape(QFrame.StyledPanel)
        self.Graph4Frame.setFrameShadow(QFrame.Raised)
        self.Graph4Text = QLabel(self.Graph4Frame)
        self.Graph4Text.setObjectName(u"Graph4Text")
        self.Graph4Text.setGeometry(QRect(10, -10, 71, 51))
        self.Graph4Text.setFont(font1)
        self.Graph4Text.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgba(255,255,255,0)")
        self.Graph3Button = QPushButton(self.page)
        self.Graph3Button.setObjectName(u"Graph3Button")
        self.Graph3Button.setGeometry(QRect(160, 450, 141, 131))
        self.Graph3Button.setStyleSheet(u"QPushButton {\n"
"background-color: #ffffff;\n"
"border: 2px solid #f66867;\n"
"	image: url(:/DownloadImage/Graphs/Graph3.jpg);\n"
"}\n"
"\n"
"")
        self.stackedWidget.addWidget(self.page)
        self.Header1.raise_()
        self.Graph3Frame.raise_()
        self.Graph1Frame.raise_()
        self.Graph2Frame.raise_()
        self.Graph1Button.raise_()
        self.Graph2Button.raise_()
        self.Graph4Frame.raise_()
        self.Graph3Button.raise_()
        self.Graph4Button.raise_()
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.Header2 = QFrame(self.page_2)
        self.Header2.setObjectName(u"Header2")
        self.Header2.setGeometry(QRect(-30, 30, 911, 91))
        self.Header2.setStyleSheet(u"background-color: #f66867")
        self.Header2.setFrameShape(QFrame.StyledPanel)
        self.Header2.setFrameShadow(QFrame.Raised)
        self.ResultText = QLabel(self.Header2)
        self.ResultText.setObjectName(u"ResultText")
        self.ResultText.setGeometry(QRect(370, 20, 121, 51))
        self.ResultText.setFont(font)
        self.ResultText.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.DownloadImage = QPushButton(self.Header2)
        self.DownloadImage.setObjectName(u"DownloadImage")
        self.DownloadImage.setGeometry(QRect(730, 64, 41, 23))
        self.DownloadImage.setStyleSheet(u"image: url(:/DownloadImage/Download.png);\n"
"background-color: rgba(250, 250, 250, 0);\n"
"border-color: rgba(250, 250, 250, 0)")
        self.ParameterButton = QPushButton(self.Header2)
        self.ParameterButton.setObjectName(u"ParameterButton")
        self.ParameterButton.setGeometry(QRect(770, 59, 41, 31))
        self.ParameterButton.setStyleSheet(u"background-color: rgba(255, 255, 255, 0);\n"
"image: url(:/DownloadImage/Parameter.png);\n"
"")
        self.QuitButton = QPushButton(self.page_2)
        self.QuitButton.setObjectName(u"QuitButton")
        self.QuitButton.setGeometry(QRect(10, 650, 101, 31))
        self.QuitButton.setStyleSheet(u"background-color: #f66867;\n"
"\n"
"")
        self.QuitImage = QFrame(self.page_2)
        self.QuitImage.setObjectName(u"QuitImage")
        self.QuitImage.setEnabled(True)
        self.QuitImage.setGeometry(QRect(10, 644, 31, 41))
        self.QuitImage.setStyleSheet(u"background-color: rgba(255, 255, 255, 0);\n"
"image: url(:/DownloadImage/Quit.png)\n"
"\n"
"\n"
"")
        self.QuitImage.setFrameShape(QFrame.StyledPanel)
        self.QuitImage.setFrameShadow(QFrame.Raised)
        self.PNGdownload = QPushButton(self.page_2)
        self.PNGdownload.setObjectName(u"PNGdownload")
        self.PNGdownload.setGeometry(QRect(800, 180, 40, 40))
        palette = QPalette()
        brush = QBrush(QColor(246, 104, 103, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        brush1 = QBrush(QColor(255, 255, 255, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        brush2 = QBrush(QColor(120, 120, 120, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.PNGdownload.setPalette(palette)
        font2 = QFont()
        font2.setFamily(u"Mangal")
        font2.setPointSize(6)
        font2.setBold(True)
        font2.setWeight(75)
        self.PNGdownload.setFont(font2)
        self.PNGdownload.setStyleSheet(u"background-color: #f66867;\n"
"border-radius:20px\n"
"")
        self.JPEGdownload = QPushButton(self.page_2)
        self.JPEGdownload.setObjectName(u"JPEGdownload")
        self.JPEGdownload.setGeometry(QRect(800, 230, 40, 40))
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.Button, brush)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush2)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.JPEGdownload.setPalette(palette1)
        self.JPEGdownload.setFont(font2)
        self.JPEGdownload.setStyleSheet(u"background-color: #f66867;\n"
"border-radius:20px\n"
"")
        self.Frame3_2 = QFrame(self.page_2)
        self.Frame3_2.setObjectName(u"Frame3_2")
        self.Frame3_2.setEnabled(True)
        self.Frame3_2.setGeometry(QRect(180, 180, 439, 419))
        self.Frame3_2.setAutoFillBackground(False)
        self.Frame3_2.setStyleSheet(u"background-color: rgb(43, 43, 58);\n"
"image: url(:/DownloadImage/graph.png);")
        self.Frame3_2.setFrameShape(QFrame.StyledPanel)
        self.Frame3_2.setFrameShadow(QFrame.Raised)
        self.stackedWidget.addWidget(self.page_2)
        self.LoadingBar = QFrame(self.centralwidget)
        self.LoadingBar.setObjectName(u"LoadingBar")
        self.LoadingBar.setGeometry(QRect(395, 670, 15, 15))
        self.LoadingBar.setStyleSheet(u"background-color: #ffffff;\n"
"border-radius: 7\n"
"")
        self.LoadingBar.setFrameShape(QFrame.StyledPanel)
        self.LoadingBar.setFrameShadow(QFrame.Raised)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
#if QT_CONFIG(tooltip)
        self.actionQuit.setToolTip(QCoreApplication.translate("MainWindow", u"Quit", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(QCoreApplication.translate("MainWindow", u"Esc", None))
#endif // QT_CONFIG(shortcut)
        self.MenuText.setText(QCoreApplication.translate("MainWindow", u"MENU", None))
        self.Graph4Button.setText("")
        self.Graph3Text.setText(QCoreApplication.translate("MainWindow", u"Graph 3", None))
        self.Graph1Text.setText(QCoreApplication.translate("MainWindow", u"Graph 1", None))
        self.Graph2Text.setText(QCoreApplication.translate("MainWindow", u"Graph 2", None))
        self.Graph1Button.setText("")
        self.Graph2Button.setText("")
        self.Graph4Text.setText(QCoreApplication.translate("MainWindow", u"Graph 4", None))
        self.Graph3Button.setText("")
        self.ResultText.setText(QCoreApplication.translate("MainWindow", u"RESULT", None))
        self.DownloadImage.setText("")
        self.ParameterButton.setText("")
        self.QuitButton.setText("")
        self.PNGdownload.setText(QCoreApplication.translate("MainWindow", u"PNG", None))
        self.JPEGdownload.setText(QCoreApplication.translate("MainWindow", u"JPEG", None))
    # retranslateUi

