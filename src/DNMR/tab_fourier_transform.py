import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

from DNMR.tab import *

class TabFourierTransform(Tab):
    def __init__(self, data_widgets, parent=None):
        super(TabFourierTransform, self).__init__(data_widgets, 'tab_ft', parent)
        
        self.data = (np.array([]), np.array([]))
        self.fit_data = np.array([])
        self.left_pivot = 1
        self.right_pivot = -1
        self.moving_left = True

    def generate_layout(self):
        #self.button_fit = QPushButton('Fit Gaussian')
        #self.button_fit.clicked.connect(self.fit)
        
        self.canvas.mpl_connect('button_press_event', self.process_button)
        
        #l = QHBoxLayout()
        #l.addWidget(self.button_fit)
        #return l

    def process_button(self, event):
        if(event.button == 1):
            if not(event.xdata is None):
                if(self.moving_left):
                    self.left_pivot = event.xdata
                else:
                    self.right_pivot = event.xdata
                self.moving_left = not(self.moving_left)
                self.update()

    def plot_logic(self):
        index = self.fileselector.spinbox_index.value()

        times = self.data_widgets['tab_phase'].data[0]
        complexes = self.data_widgets['tab_phase'].data[1]

        timespacing = (times[index][1]-times[index][0])
    
        time_index = np.argmin(np.abs(self.fileselector.data['peak_locations'][:,None] - times), axis=1)
        s_complexes = np.zeros_like(complexes)
        s_times = np.zeros_like(times)
        for i in range(s_complexes.shape[0]):
            s_complexes[i,:] = np.roll(complexes[i,:], -time_index[i])
            s_times[i,:]     = np.roll(times[i,:],     -time_index[i])
        s_reals = np.real(s_complexes)
        s_imags = np.imag(s_complexes)

        fftfreq = np.fft.fftshift(np.fft.fftfreq(s_complexes[index].shape[0], d=timespacing)) # microseconds
        fft = np.zeros_like(s_complexes, dtype=np.complex128)
        for i in range(s_complexes.shape[0]):
            fft[i] = np.fft.fftshift(np.fft.fft(s_complexes[i]))
        self.data = (fftfreq, fft)
        
        self.ax.plot(fftfreq, np.real(fft[index]), 'r', alpha=0.6, label='R')
        self.ax.plot(fftfreq, np.imag(fft[index]), 'b', alpha=0.6, label='I')
        self.ax.plot(fftfreq, np.abs(fft[index]), 'k', alpha=0.3, label='abs')
        
        if(self.fit_data.shape[0] > 0):
            self.ax.plot(fftfreq, self.fit_data, 'k--', alpha=0.5, label='fit')
        
        self.ax.set_xlabel('frequency (MHz)')

        self.ax.axvline(self.left_pivot, color='k')
        self.ax.axvline(self.right_pivot, color='k')

    def fit(self):
        '''Fits a gaussian'''
        def gauss(args, x):
            return args[0]*np.exp(-np.square((x-args[2])/(2*args[1]))) + args[3]
        
        res = sp.optimize.minimize(lambda args: np.sum(np.square(np.abs(self.data[1]) - gauss(args, self.data[0]))), 
                                   x0=[np.max(np.abs(self.data[1])),0.1,0.0,np.std(np.abs(self.data[1]))],
                                   bounds=[[0, np.max(np.abs(self.data[1]))*2],[0, np.max(self.data[0])/2.0],[0, np.max(self.data[0])/2.0],[0,np.max(np.abs(self.data[1]))*2]],
                                   method='Nelder-Mead')
        print(res)
        self.fit_data = gauss(res.x, self.data[0])
        self.update()
        
    def get_exported_data(self):
        index = self.fileselector.spinbox_index.value()
        return { 'times': self.data_widgets['tab_phase'].data[0][index],
                 'complexes': self.data_widgets['tab_phase'].data[1][index],
                 'frequencies (MHz)': self.data[0],
                 'fft': self.data[1][index],
               }