import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
import numpy as np

import DNMR.fileops as fileops

class PhaseAdjustmentWidget(QWidget):
    def __init__(self, parent=None, callback=lambda: None):
        super(PhaseAdjustmentWidget, self).__init__(parent)

        self.slider_phase = QSlider()
        self.slider_phase.setRange(-180, 180)
        self.slider_phase.valueChanged.connect(callback)
        self.slider_phase.valueChanged.connect(lambda v: self.label_phase.setText(f'Phase: {v}\u00b0'))
        self.slider_phase.setOrientation(Qt.Orientation.Horizontal)
        self.label_phase = QLabel('Phase: 0\u00b0')

        layout = QVBoxLayout()
        layout.addWidget(self.label_phase)
        layout.addWidget(self.slider_phase)
        self.setLayout(layout)

class FileInfoWidget(QWidget):
    def __init__(self, parent=None):
        super(FileInfoWidget, self).__init__(parent)
        
        self.listview_docinfo = QListWidget()
        
        layout = QVBoxLayout()
        layout.addWidget(self.listview_docinfo)
        self.setLayout(layout)
        
    def update_items(self, d, length=None, prefix=''):
        '''Takes a data_struct'''
        if(length is None):
            length = d['size']
        for i in list(d.keys()):
            if(isinstance(d[i], fileops.data_struct)):
                self.update_items(d[i], length=length, prefix=prefix+str(i)+'/')
            elif(isinstance(d[i], np.ndarray)):
                if(d[i].ndim == 1):
                    s = '\n'.join([ f'\t{j}: ' + str(d[i][j]) for j in range(length) ])
                    self.listview_docinfo.addItem(f'{prefix+i} (array, len={d[i].shape[0]})='+'{\n'+s+'\n}')
                elif(d[i].ndim == 2):
                    # first index is scan index, second is datapoint
                    s = '\n'.join([ f'\t{j}: ' + str(d[i][j]) for j in range(length) ])
                    self.listview_docinfo.addItem(f'{prefix+i} (array, len={d[i].shape[0]}x{d[i].shape[1]})='+'{\n'+s+'\n}')
            else:
                self.listview_docinfo.addItem(f'{prefix+i}={d[i]}')
                
                    

class FileSelectionWidget(QWidget):
    def __init__(self, parent=None):
        super(FileSelectionWidget, self).__init__(parent)
        
        self.filedialog = QFileDialog()
        self.button_load = QPushButton('Load')
        self.button_load.clicked.connect(self.open_file)
        self.button_info = QPushButton('Info')
        self.button_info.clicked.connect(self.file_info)
        self.label_index = QLabel('Index:')
        self.spinbox_index = QSpinBox()
        self.label_channel = QLabel('Channel:')
        self.spinbox_channel = QSpinBox()
        self.checkbox_holdplots = QCheckBox('Hold plots')

        layout = QHBoxLayout()
        l0 = QVBoxLayout()
        l0.addWidget(self.button_load)
        l0.addWidget(self.button_info)
        layout.addLayout(l0)
        l = QHBoxLayout()
        l.addWidget(self.label_index)
        l.addWidget(self.spinbox_index)
        l2 = QHBoxLayout()
        l2.addWidget(self.label_channel)
        l2.addWidget(self.spinbox_channel)
        l_m = QVBoxLayout()
        l_m.addLayout(l)
        l_m.addLayout(l2)
        layout.addLayout(l_m)
        #layout.addWidget(self.spinbox_index)
        layout.addWidget(self.checkbox_holdplots)
        self.setLayout(layout)

        self.fn = ''
        self._fn = [''] # for all channels
        self.data = {}
        self._data = [{}] # for all channels
        
        self.infodialogs = []

        self.callbacks = []
        self.spinbox_index.valueChanged.connect(self.callback)
        self.spinbox_channel.valueChanged.connect(self.channel_callback)
    
    def channel_callback(self):
        while(len(self._fn) <= self.spinbox_channel.value()):
            self._fn += ['']
            self._data += [{}]
        self.fn = self._fn[self.spinbox_channel.value()]
        self.data = self._data[self.spinbox_channel.value()]
        if(self.fn != ''):
            self.spinbox_index.setRange(0, self.data['size']-1)
        self.spinbox_index.setValue(0)
        #self.callback()
    
    def callback(self):
        for i in self.callbacks:
            i()

    def open_file(self):
        try:
            fns = self.filedialog.getOpenFileNames()[0]
            big_data = fileops.get_data(fns[0])
            if(len(fns) > 1):
                for fn in fns[1:]:
                    data = fileops.get_data(fn)
                    big_data += data
            self.fn = fns[0] # above lines will throw exceptions if anything bad happens, so if anything bad happens, we want to preserve the previous file being loaded
            self.data = big_data
            self._fn[self.spinbox_channel.value()] = self.fn
            self._data[self.spinbox_channel.value()] = self.data
            self.spinbox_index.setRange(0, self.data['size']-1)
            self.spinbox_index.setValue(0)
            self.callback()
        except Exception as e:
            traceback.print_exc()
            
    def file_info(self):
        if(len(list(self.data.keys())) != 0):
            self.infodialogs += [FileInfoWidget()] # must be stored somewhere, or python garbage collection will clean it up and close the window
            self.infodialogs[-1].update_items(self.data)
            self.infodialogs[-1].setWindowTitle(f'Info on {self._fn[self.spinbox_channel.value()]}')
            self.infodialogs[-1].show()
    