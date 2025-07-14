
import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

from DNMR.miniwidgets import *
from DNMR.tab import Tab

class TabPhaseAdjustment(Tab):
    def __init__(self, data_widgets, parent=None):
        super(TabPhaseAdjustment, self).__init__(data_widgets, 'tab_phase', parent)
        
        self.data = (np.array([]), np.array([]))

    def generate_layout(self):
        self.phase_adjustment = PhaseAdjustmentWidget(callback=self.update_phase)
        self.data_widgets['phase_adjustment'] = self.phase_adjustment
        self.canvas.mpl_connect('button_press_event', self.process_button)
        
        self.spinbox_filtersize = QSpinBox()
        self.spinbox_filtersize.setRange(0, 100)
        self.spinbox_filtersize.setValue(12)
        self.spinbox_filtersize.valueChanged.connect(self.update)
        
        self.combobox_filtertype = QComboBox()
        self.combobox_filtertype.addItems(['Sinc','Gaussian','Half-Gaussian','Median','None'])
        self.combobox_filtertype.currentTextChanged.connect(self.update)
        
        self.pushbutton_phaseadjust = QPushButton('Autophase')
        self.pushbutton_phaseadjust.clicked.connect(self.autophase)
        
        self.pushbutton_applyall = QPushButton('Apply to all')
        self.pushbutton_applyall.clicked.connect(lambda: self.phase_set(self.phase_adjustment.slider_phase.value()))

        l2 = QVBoxLayout()
        l0 = QHBoxLayout()
        l0.addWidget(self.phase_adjustment)
        l3 = QVBoxLayout()
        l3.addWidget(self.pushbutton_phaseadjust)
        l3.addWidget(self.pushbutton_applyall)
        l0.addLayout(l3)
        l1 = QHBoxLayout()
        l1.addWidget(self.combobox_filtertype)
        l1.addWidget(self.spinbox_filtersize)
        l2.addLayout(l0)
        l2.addLayout(l1)
        return l2

    def autophase(self):
        if(self.data[0].shape[0] == 0):
            return
        
        # find peak, current phase at peak
        index = self.fileselector.spinbox_index.value()
        times = self.data[0][index]
        complexes = self.data[1][index]
        mags = np.abs(complexes)
        
        peak_i = np.argmax(mags)
        peak_t = times[peak_i]
        peak_c = complexes[peak_i]
        angle = np.arctan2(np.imag(peak_c),np.real(peak_c))
        angle_degrees = angle * 180.0/np.pi
        
        current_phases = self.get_global_phaseset() # degrees
        new_phase_degrees = current_phases[index]-angle_degrees
        # modulo 180
        new_phase_degrees += (np.abs(new_phase_degrees - 180.0)//360)*360.0
        new_phase_degrees -= (np.abs(new_phase_degrees + 180.0)//360)*360.0
        
        # now do the setting phase to zero.
        self.fileselector.data['phases'] = [ new_phase_degrees for i in range(len(self.fileselector.data['phases'])) ]
        self.get_global_peaklocs()
        self.fileselector.data['peak_locations'] = np.ones_like(self.get_global_peaklocs()) * peak_t
        
        self.update()

    def get_global_phaseset(self):
        if('phases' in self.fileselector.data.keys()):
            ps = self.fileselector.data['phases']
        else:
            self.fileselector.data['phases'] = [0 for i in range(self.fileselector.data['size'])]
            ps = self.fileselector.data['phases']
        return ps

    def get_global_peaklocs(self):
        if('peak_locations' in self.fileselector.data.keys()):
            ps = self.fileselector.data['peak_locations']
        else:
            self.fileselector.data['peak_locations'] = np.array([0 for i in range(self.fileselector.data['size'])])
            ps = self.fileselector.data['peak_locations']
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
        peak_locs = self.get_global_peaklocs() # Make sure it exists.
        if(event.button == 1):
            if not(event.xdata is None):
                self.fileselector.data['peak_locations'] = np.ones_like(self.get_global_peaklocs()) * event.xdata
            self.update()
        elif(event.button == 3): # why set two or three indices, when you could set **just one**?
            index = self.fileselector.spinbox_index.value()
            if not(event.xdata is None):
                self.fileselector.data['peak_locations'][index] = event.xdata
            self.update()

    def plot_logic(self):
        index = self.fileselector.spinbox_index.value()
        reals = self.fileselector.data.reals
        imags = self.fileselector.data.imags
        times = self.fileselector.data.times
        times = times[:,:reals.shape[1]]
        
        ps = self.get_global_phaseset()
        self.phase_adjustment.slider_phase.setValue(int(ps[index]))

        complexes = reals + 1j*imags
        
        for i in range(complexes.shape[0]):
            t = self.combobox_filtertype.currentText()
            if(t == 'Gaussian'):
                kernel = np.exp(-1/2 * np.square(np.linspace(-3, 3, self.spinbox_filtersize.value()*2+1)))
            elif(t == 'Sinc'):
                kernel = np.sinc(np.linspace(-reals.shape[1]/(2*self.spinbox_filtersize.value() + 1), reals.shape[1]/(2*self.spinbox_filtersize.value() + 1), reals.shape[1]))
            elif(t == 'Half-Gaussian'):
                i_s = np.linspace(-3, 3, self.spinbox_filtersize.value()*2+1)
                kernel = np.where(i_s >= 0, np.exp(-1/2 * np.square(i_s)), 0)
            elif(t == 'None'):
                kernel = 1
            elif(t == 'Median'):
                complexes[i] = sp.ndimage.median_filter(reals[i], mode='wrap', size=self.spinbox_filtersize.value()) + 1j * sp.ndimage.median_filter(imags[i], mode='wrap', size=self.spinbox_filtersize.value())
                continue
            kernel /= np.sum(kernel)
            complexes[i] = np.convolve(complexes[i], kernel, mode='same')

        phases = np.exp(1j*np.array(ps)* np.pi/180.0)
        complexes *= phases[:,None]

        #complexes -= np.average(complexes, axis=1)[:,None]

        peak_loc = self.get_global_peaklocs()[index]
        self.ax.axvline(peak_loc, color='k', linestyle='--', alpha=0.5)

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