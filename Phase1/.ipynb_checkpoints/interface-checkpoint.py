#==========================================
# User Interface 
# Author: Ali Salman
# Date:   Dec. 2020
#==========================================

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

#import threading
from multiprocessing import  Queue
import time
import sounddevice as sd
import soundfile as sf

class AudioPlayer(QWidget):
    
    def __init__(self):
        super().__init__()
        self.audios = [None, None]
        self.fs = [None, None]
        self.set = [False, False]
        self.response = Queue()
        
        self.initUI()
             
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 12))
        
        self.setToolTip('This is a super cool audio interface')
        
        btn1 = QPushButton('Play Audio 1', self)
        btn1.setToolTip('Plays the first audio file')
        btn1.clicked.connect(self.play_audio(0))
        btn1.resize(btn1.sizeHint())
        btn1.move(50, 50)  
        
        btn2 = QPushButton('Play Audio 2', self)
        btn2.setToolTip('Plays the second audio file')
        btn2.clicked.connect(self.play_audio(1))
        btn2.resize(btn2.sizeHint())
        btn2.move(250, 50)
        
        btn3 = QPushButton('Stop Audio 1', self)
        btn3.setToolTip('Stop the first audio file')
        btn3.clicked.connect(self.stop_audio(0))
        btn3.resize(btn3.sizeHint())
        btn3.move(50, 80)
        
        btn3 = QPushButton('Stop Audio 1', self)
        btn3.setToolTip('Stop the first audio file')
        btn3.clicked.connect(self.stop_audio(0))
        btn3.resize(btn3.sizeHint())
        btn3.move(250, 80)
        
        self.btn5 = QPushButton('Select audio 1', self)
        self.btn5.setToolTip('Plays the first audio file')
        self.btn5.clicked.connect(self.myprint(0))
        self.btn5.resize(self.btn5.sizeHint())
        self.btn5.move(50, 120)  
        
        self.btn6 = QPushButton('Select audio 2', self)
        self.btn6.setToolTip('Plays the first audio file')
        self.btn6.clicked.connect(self.myprint(1))
        self.btn6.resize(self.btn6.sizeHint())
        self.btn6.move(250, 120)
        
        self.btn7 = QPushButton('Both', self)
        self.btn7.setToolTip('Plays the first audio file')
        self.btn7.clicked.connect(self.myprint(2))
        self.btn7.resize(self.btn7.sizeHint())
        self.btn7.move(50, 150)
        
        self.btn8 = QPushButton('Neither', self)
        self.btn8.setToolTip('Plays the first audio file')
        self.btn8.clicked.connect(self.myprint(3))
        self.btn8.resize(self.btn8.sizeHint())
        self.btn8.move(250, 150)
        
        #         btn_grp = QButtonGroup()
        #         btn_grp.setExclusive(True)
        #         btn_grp.addButton(self.btn3)
        #         btn_grp.addButton(self.btn4)
        #         btn_grp.clicked.connect(self.myprint)
        
        self.disable_buttons()
        self.setGeometry(300, 300, 400, 400)
        self.setWindowTitle('Super cool audio interface')
        #self.setWindowIcon(QIcon('web.png'))        
        self.center()
        self.show()
        
    def myprint(self, num):
        def player():
            self.response.put(num)
            self.btn5.setDisabled(False)
            self.btn6.setDisabled(False)
        return player
    
    
    def disable_buttons(self):
        self.btn5.setDisabled(True)
        self.btn6.setDisabled(True)
        self.btn7.setDisabled(True)
        self.btn8.setDisabled(True)
        
    def enable_buttons(self):
        self.btn5.setDisabled(False)
        self.btn6.setDisabled(False)
        self.btn7.setDisabled(False)
        self.btn8.setDisabled(False)
        
    #puts the window in the wasat
    def set_audio(self, audio, fs, num):
        self.audios[num] = audio
        self.fs[num] = fs
        self.set[num] = True
        if all(self.set):
            self.enable_buttons()
        

    def play(self, num):
        if self.set[num]:
            sd.play(self.audios[num], self.fs[num])
            #status = sd.wait()
            
    def play_audio(self, num):
        def player():
            self.play(num)
        return player
    
    def stop(self, num):
        if self.set[num]:
            sd.stop()
    
    def stop_audio(self,num):
        def stopper():
            self.stop(num)
        return stopper
            
    def get_response(self):
        while self.response.empty():
            qApp.processEvents()
            time.sleep(0.05)
            
        tmp = self.response.get()
        self.disable_buttons()
        return tmp
    
    def center(self):
        
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

