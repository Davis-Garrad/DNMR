
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

class TabPhaseAdjustment(Tab):
    def __init__(self, data_widgets, parent=None):
        super(TabPhaseAdjustment, self).__init__(data_widgets, 'tab_phase', parent)
        
        self.data = (np.array([]), np.array([]))
        self.pivot_location = 0

    def generate_layout(self):
        self.phase_adjustment = PhaseAdjustmentWidget(callback=self.update_phase)
        self.data_widgets['phase_adjustment'] = self.phase_adjustment
        self.canvas.mpl_connect('button_press_event', self.process_button)
        
        self.spinbox_filtersize = QSpinBox()
        self.spinbox_filtersize.setRange(0, 100)
        self.spinbox_filtersize.setValue(12)
        self.spinbox_filtersize.valueChanged.connect(self.update)
        
        self.pushbutton_applyall = QPushButton('Apply to all')
        self.pushbutton_applyall.clicked.connect(lambda: self.phase_set(self.phase_adjustment.slider_phase.value()))

        l = QHBoxLayout()
        l.addWidget(self.phase_adjustment)
        l.addWidget(self.pushbutton_applyall)
        l.addWidget(self.spinbox_filtersize)
        return l

    def get_global_phaseset(self):
        if('phases' in self.fileselector.data.keys()):
            ps = self.fileselector.data['phases']
        else:
            self.fileselector.data['phases'] = [0 for i in range(self.fileselector.data['size'])]
            ps = self.fileselector.data['phases']
        return ps
    
    def phase_set(self, p):
        ps = self.get_global_phaseset()
        for i in range(len(ps)):
            ps[i] = p
        self.update()

    def update_phase(self):
        ps = self.get_global_phaseset()
        index = self.fileselector.spinbox_index.value()
        ps[index] = self.phase_adjustment.slider_phase.value()
        self.update()

    def process_button(self, event):
        if(event.button == 1):
            if not(event.xdata is None):
                self.pivot_location = event.xdata
            self.update()

    def plot_logic(self):
        index = self.fileselector.spinbox_index.value()
        reals = self.fileselector.data.reals
        imags = self.fileselector.data.imags
        times = self.fileselector.data.times
        times = times[:,:reals.shape[1]]
        
        ps = self.get_global_phaseset()
        self.phase_adjustment.slider_phase.setValue(ps[index])

        complexes = reals + 1j*imags
        
        for i in range(complexes.shape[0]):
            kernel = np.exp(-1/2 * np.square(np.linspace(-3, 3, self.spinbox_filtersize.value()*2+1)))
            kernel /= np.sum(kernel)
            complexes[i] = np.convolve(complexes[i], kernel, mode='same')
        #    complexes[i] = sp.ndimage.median_filter(reals[i], mode='wrap', size=25) + 1j * sp.ndimage.median_filter(imags[i], mode='wrap', size=25)

        phases = np.exp(1j*np.array(ps)* np.pi/180.0)
        complexes *= phases[:,None]

        #complexes -= np.average(complexes, axis=1)[:,None]

        self.ax.axvline(self.pivot_location, color='k', linestyle='--', alpha=0.5)

        total = np.sum(np.square(np.abs(complexes[index]))) * (times[index][1] - times[index][0])

        reals = np.real(complexes)
        imags = np.imag(complexes)

        self.ax.plot(times[index][:reals[index].shape[0]], np.abs(complexes[index]), 'k', alpha=0.6, label=f'Mag.')
        self.ax.plot(times[index][:reals[index].shape[0]], reals[index], 'r', alpha=0.6, label='R')
        self.ax.plot(times[index][:imags[index].shape[0]], imags[index], 'b', alpha=0.6, label='I')

        self.ax.plot(times[index][:reals[index].shape[0]], np.average(reals, axis=0), 'r--', alpha=0.3, label='Avg. R')
        self.ax.plot(times[index][:imags[index].shape[0]], np.average(imags, axis=0), 'b--', alpha=0.3, label='Avg. I')
        self.ax.set_xlabel('time (us)')

        self.ax.axhline(0, color='k', linestyle='-')

        self.data = (times[:][:reals[index].shape[0]], complexes)