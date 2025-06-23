
import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

import TNMRPlot.fileops as fileops
from TNMRPlot.miniwidgets import *

from TNMRPlot.tab import Tab

class TabT1Fit(Tab):
    output_frames = {}

    def __init__(self, data_widgets, parent=None):
        super(TabT1Fit, self).__init__(data_widgets, 'tab_t1_fitting', parent)
        
        self.data = (np.array([]), np.array([]))
        self.plot_data = (np.array([]), np.array([]))
        self.x0 = None

    def generate_layout(self):
        self.combobox_fittingroutine = QComboBox()
        self.combobox_fittingroutine.addItem('7/2 Spin')
        
        self.pushbutton_fit = QPushButton('Fit')
        self.pushbutton_fit.clicked.connect(self.fit)
        
        self.checkbox_normalize = QCheckBox('Normalize?')

        l = QHBoxLayout()
        lv = QVBoxLayout()
        lv.addWidget(self.combobox_fittingroutine)
        lv.addWidget(self.checkbox_normalize)
        l.addLayout(lv)
        l.addWidget(self.pushbutton_fit)

        # fit output
        seven_halves_frame = QFrame() # TODO: Make this better.
        seven_halves_frame.hide()
        lo = QVBoxLayout()
        a = QLabel('fitting...')
        b = QLabel('fitting...')
        c = QLabel('fitting...')
        d = QLabel('fitting...')
        lo.addWidget(a)
        lo.addWidget(b)
        lo.addWidget(c)
        lo.addWidget(d)
        seven_halves_frame.setLayout(lo)
        self.output_frames['7/2 Spin'] = [ seven_halves_frame, {'widget':a, 'label':'y0'}, {'widget': b, 'label': 's'}, {'widget': c, 'label': 'T1'}, {'widget': d, 'label': 'r'} ]

        l.addWidget(seven_halves_frame)

        return l

    def plot_logic(self):
        freq = self.data_widgets['tab_ft'].data[0]
        ft   = self.data_widgets['tab_ft'].data[1]
        real = np.real(ft)

        #for i in range(real.shape[0]):
        #    kernel = np.exp(-1/2 * np.square(np.linspace(-3, 3, 15)))
        #    kernel /= np.sum(kernel)
        #    real[i] = np.convolve(real[i], kernel, mode='same')

        #self.ax.plot(real[0], label='filtered')

        integrations = np.zeros(real.shape[0])
        start_index = np.argmin(np.abs(self.data_widgets['tab_ft'].left_pivot - freq))
        end_index = np.argmin(np.abs(self.data_widgets['tab_ft'].right_pivot - freq))
        if(end_index < start_index):
            tmp = start_index
            start_index = end_index
            end_index = tmp

        integrations = np.sum(real[:,start_index:end_index], axis=1)
        if(self.checkbox_normalize.isChecked()):
            integrations /= np.max(integrations)
        
        self.ax.set_xscale('log')
        self.ax.set_xlabel('relaxation time (us)')
        self.ax.plot(self.fileselector.data.relaxation_times, integrations, label='integrations', linestyle='', marker='o')

        self.data = (self.fileselector.data.relaxation_times, integrations)

        if(self.plot_data[0].shape[0] > 0):
            self.ax.plot(self.plot_data[0], self.plot_data[1], label='fit')
        
    def fit(self):
        bounds = None
        for key, val in self.output_frames.items():
            val[0].hide()
        out_frame = self.output_frames[self.combobox_fittingroutine.currentText()]
        out_frame[0].show()

        if(self.combobox_fittingroutine.currentText() == '7/2 Spin'):
            #bounds = [ [ -np.inf, np.inf ], [-1, 10], [np.min(self.fileselector.rel_times), np.max(self.fileselector.rel_times)*10], [-1.2, 1.2] ]
            bounds = [ [0, np.max(self.data[1])*10], [-1, 10], [np.min(self.fileselector.data.relaxation_times)/10, np.max(self.fileselector.data.relaxation_times)*10], [0.99, 1.01] ]
            def fit_func(args, x):
                gamma_0 = args[0]
                s = args[1] # inversion
                T1 = args[2] # relaxation time (actual fit variable, really)
                r = args[3] # stretched exponent (ideally 1)
                #y = y0 (1-(1+s) ((1/84)*Exp[-(t/T1)^r]+(3/44)*Exp[-(6 t/T1)^r]+(75/364)*Exp[-(15 t/T1)^r]+(1225/1716)*Exp[-(28 t/T1)^r]))
                fit = gamma_0 * (1-(1+s)*(
                                            (1/84)*     np.exp(-np.pow(x/T1,    r)) + 
                                            (3/44)*     np.exp(-np.pow(6*x/T1,  r)) +
                                            (75/364)*   np.exp(-np.pow(15*x/T1, r)) +
                                            (1225/1716)*np.exp(-np.pow(28*x/T1, r)) 
                                         ))
                return fit
        def cost_func(args, x, y):
            #gaps = np.zeros_like(x[:-1]) # for points that are much closer together, it matters proportionally less that all of them are perfectly fit as they each contribute to the cost
            #gaps = x[1:] - x[:-1]
            #dx = np.zeros_like(x)
            #dx[1:] += gaps/2.0
            #dx[:-1] += gaps/2.0
            #dx /= dx
            # essentially, minimize the integral of the squared differences

            #return np.sum(np.square((fit_func(args, x) - y) * dx))
            return np.sum(np.square((fit_func(args, x) - y)))

        #popt, pcov = sp.optimize.curve_fit(fit_func, self.data[0][10:], self.data[1][10:], bounds=bounds)
        #res = sp.optimize.minimize(lambda x: cost_func(x, self.data[0], self.data[1]), x0=[0]*4 if self.x0 is None else self.x0, method='Nelder-Mead', bounds=bounds)
        if(self.checkbox_normalize.isChecked()):
            d0 = self.data[0]
            d1 = self.data[1] / np.max(self.data[1])
            self.data = (d0, d1)
        res = sp.optimize.differential_evolution(lambda x: cost_func(x, 
                                                                     self.data[0], 
                                                                     self.data[1]), 
                                                 bounds=bounds)
        #res = sp.optimize.brute(lambda x: cost_func(x, self.data[0], self.data[1]), bounds)
        print(res)
        self.x0 = res.x
        self.plot_data = (self.data[0], fit_func(res.x, self.data[0]))
        for i in range(len(self.x0)):
            out_frame[i+1]['widget'].setText(f'{out_frame[i+1]["label"]}={self.x0[i]}')
        self.update()