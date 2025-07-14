
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
from DNMR.fileops import data_struct

class TabPeakAmplitude(Tab):
    def __init__(self, data_widgets, parent=None):
        super(TabPeakAmplitude, self).__init__(data_widgets, 'tab_peakamp', parent)
        
        self.fileselector.callbacks += [ self.retrieve_labels ]
        
    def generate_layout(self):
        self.combobox_labelling = QComboBox()
        self.combobox_labelling.addItem('Load Order')
        self.combobox_labelling.setCurrentIndex(0)
        
        layout = QHBoxLayout()
        layout.addWidget(self.combobox_labelling)
        return layout
        
    def deconstruct_dict(self, d, plunge=False):
        keys = []
        for k in d.keys():
            try:
                if(isinstance(d[k], data_struct)):
                    ks = self.deconstruct_dict(d[k], True)
                    for i in range(len(ks)):
                        ks[i] = f'{k}/{ks[i]}'
                    keys += ks
                elif(plunge):
                    keys += [k]
                    raise Exception
                a = d[k][0] # iterable
                if(len(d[k]) == self.fileselector.data['size']): # good size
                    good = False
                    try:
                        a = a[0]  # not >1D
                        if(len(d[k][0]) == 1):
                            good = True # only pseudo->1D
                    except:
                        good = True
                    if(good):
                        if(isinstance(d[k][0], data_struct)):
                            ks = self.deconstruct_dict(d[k][0], True)
                            for i in range(len(ks)):
                                ks[i] = f'{k}/{ks[i]}'
                            keys += ks
                        else:
                            keys += [k]
            except:
                pass
        return keys
        
    def retrieve_labels(self):
        for i in range(self.combobox_labelling.count()):
            self.combobox_labelling.removeItem(0)
            
        self.combobox_labelling.addItem('Load Order')
        keys = self.deconstruct_dict(self.fileselector.data)
        for k in keys:
            self.combobox_labelling.addItem(k)

    def plot_logic(self):
        
        freq = self.data_widgets['tab_ft'].data[0]
        ft   = self.data_widgets['tab_ft'].data[1]
        real = np.real(ft)
        try:
            del_times = self.fileselector.data.sequence['0'].delay_time
        except:
            del_times = self.fileselector.data.sequence['0'].relaxation_time # Legacy

        integrations = np.zeros(real.shape[0])
        start_index = np.argmin(np.abs(self.data_widgets['tab_ft'].left_pivot - freq))
        end_index = np.argmin(np.abs(self.data_widgets['tab_ft'].right_pivot - freq))
        if(end_index < start_index):
            tmp = start_index
            start_index = end_index
            end_index = tmp

        integrations = np.sum(ft[:,start_index:end_index], axis=1)
        integrations /= np.max(np.abs(integrations))
        integrals = integrations

        if(self.combobox_labelling.currentText() == 'Load Order'):
            indices = np.linspace(0, self.fileselector.data['size'], self.fileselector.data['size'], endpoint=False)
        else:
            k = self.combobox_labelling.currentText()
            ks = k.split('/')
            d0 = self.fileselector.data[ks[0]]
            for K in ks[1:]:
                d0 = d0[K]
            indices = d0
            indices = np.reshape(indices, shape=integrals.shape)
        self.ax.set_xlabel(self.combobox_labelling.currentText())
        sorter = np.argsort(indices)
        indices = indices[sorter]

        reals = np.real(integrals)
        imags = np.imag(integrals)
        mags = np.abs(integrals)
        
        self.ax.plot(indices, mags[sorter], 'k', alpha=0.6, label=f'Mag.')
        self.ax.plot(indices, reals[sorter], 'r', alpha=0.6, label='R')
        self.ax.plot(indices, imags[sorter], 'b', alpha=0.6, label='I')

