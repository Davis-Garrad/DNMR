
import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

from TNMRPlot.miniwidgets import *
from TNMRPlot.tab import Tab

class TabFieldScan(Tab):
    def __init__(self, data_widgets, parent=None):
        super(TabFieldScan, self).__init__(data_widgets, 'tab_fieldscan', parent)

    def plot_logic(self):
        if('ppms_mf' in self.fileselector.data.keys()):
            index = self.fileselector.spinbox_index.value()

            times = self.data_widgets['tab_phase'].data[0]
            complexes = self.data_widgets['tab_phase'].data[1]

            reals = np.real(complexes)
            imags = np.imag(complexes)
            
            real_integral = np.sum(reals, axis=1)
            imag_integral = np.sum(imags, axis=1)
            mag_integral = np.sum(np.abs(reals + 1j*imags), axis=1)

            fields = self.fileselector.data.ppms_mf

            self.ax.plot(fields, np.abs(real_integral + 1j*imag_integral), 'k', alpha=0.6, label=f'Mag.')
            self.ax.plot(fields, real_integral, 'r', alpha=0.6, label='R')
            self.ax.plot(fields, imag_integral, 'b', alpha=0.6, label='I')
            
            self.ax.set_xlabel('field (T)')
            
            
            pvt = self.data_widgets['tab_phase'].pivot_location
            max_index = np.argmin(np.abs(times[0] - pvt))
            r = reals[:,max_index]
            i = imags[:,max_index]
            M = np.sqrt(np.square(r) + np.square(i))
            self.ax.plot(fields, M, linestyle='None', marker='o', color='k')
            self.ax.plot(fields, r, linestyle='None', marker='o', color='r')
            self.ax.plot(fields, i, linestyle='None', marker='o', color='b')

