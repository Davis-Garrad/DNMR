
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

class TabInvLaplace(Tab):
    output_frames = {}

    def __init__(self, data_widgets, parent=None):
        super(TabInvLaplace, self).__init__(data_widgets, 'tab_inv_laplace', parent)
        
        self.plotted_data = []
        
    def generate_layout(self):
        l = QHBoxLayout()
        p = QPushButton('Fit')
        p.clicked.connect(self.fit)
        l.addWidget(p)
        return l

    def plot_logic(self):
        for i in self.plotted_data:
            
            self.ax.plot(i[0], i[1], label=i[2])
            
    def fit(self):
        ts = self.data_widgets['tab_phase'].data[0]
        freq = self.data_widgets['tab_ft'].data[0]
        ft   = self.data_widgets['tab_ft'].data[1]
        imag = np.imag(ft)
        real = np.real(ft)
        F = real + 1j*imag

        integrations = np.zeros(real.shape[0])
        start_index = np.argmin(np.abs(self.data_widgets['tab_ft'].left_pivot - freq))
        end_index = np.argmin(np.abs(self.data_widgets['tab_ft'].right_pivot - freq))
        if(end_index < start_index):
            tmp = start_index
            start_index = end_index
            end_index = tmp

        integrations = np.sum(F[:,start_index:end_index], axis=1)
        try:
            ts = self.fileselector.data.sequence['0'].delay_time
        except:
            ts = self.fileselector.data.sequence['0'].relaxation_time # Legacy
        
        num_bins = 250
        T1s = np.exp(np.linspace(np.log(1e5), np.log(1e7), num_bins))
        inv_T1s = 1/T1s
        
        # for 7/2 spin.
        qs = np.array([1,6,15,28])
        ps = np.array([1/84, 3/44, 75/364, 1225/1716])
        print(ts)
        kernel = np.sum(1 - 2*ps[:,None,None]*np.exp(-qs[:,None,None] * ts[None,:,None]/T1s[None,None,:]), axis=0) # K[i,j]
        kernel = np.matrix(kernel)
        
        def cost_function(M, K, P, alpha):
            return np.square(np.linalg.norm(M - K@P)) + alpha*np.square(np.linalg.norm(P))
            
        bounds = [ [ 1e-9, 1.0] for i in range(num_bins) ]
        # WAY UNDERDETERMINED
        self.plotted_data = []
        for a in [1e0, 1e1, 1e2, 1e3, 1e4]:
            #res = sp.optimize.differential_evolution(lambda x, *args: cost_function(args[0], args[1], x, a), bounds, args=(integrations, kernel), constraints=(sp.optimize.LinearConstraint(np.identity(num_bins), 1.0, 1.01),))
            x0 = np.exp(-np.square(np.linspace(-10, 10, num_bins))/2)
            res = sp.optimize.minimize(lambda x, *args: cost_function(args[0], args[1], x, a), x0=x0, args=(integrations, kernel), bounds=bounds, method='SLSQP', constraints=({'type': 'eq', 'fun': lambda x: 1-sum(x)},))
            #res = sp.optimize.direct(lambda x, *args: cost_function(args[0], args[1], x, a), args=(integrations, kernel), bounds=bounds)
            normed = res.x / np.sum(res.x)
            self.plotted_data += [(T1s, normed, f'alpha={a}')]
            print(res)
        
        self.update()
        