
import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

import DNMR.fileops as fileops
from DNMR.miniwidgets import *

from DNMR.tab import Tab

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
            
            self.ax.plot(i[0], i[1], label=i[2], alpha=0.5)
            
    def fit(self):
        ts = self.data_widgets['tab_phase'].data[0]
        freq = self.data_widgets['tab_ft'].data[0]
        ft   = self.data_widgets['tab_ft'].data[1]
        imag = np.imag(ft)
        real = np.real(ft)
        F = real + 1j*imag
        
        # for 7/2 spin.
        qs = np.array([1,6,15,28])
        ps = np.array([1/84, 3/44, 75/364, 1225/1716])
        num_bins = 250
        T1s = np.exp(np.linspace(np.log(4.5e5), np.log(5.5e5), num_bins))

        integrations = np.zeros(real.shape[0])
        start_index = np.argmin(np.abs(self.data_widgets['tab_ft'].left_pivot - freq))
        end_index = np.argmin(np.abs(self.data_widgets['tab_ft'].right_pivot - freq))
        if(end_index < start_index):
            tmp = start_index
            start_index = end_index
            end_index = tmp

        integrations = np.sum(F[:,start_index:end_index], axis=1)
        integrations -= np.min(integrations)
        integrations /= np.max(integrations)
        integrations *= 2.0
        integrations -= 1.0
        try:
            ts = self.fileselector.data.sequence['0'].delay_time
        except:
            ts = self.fileselector.data.sequence['0'].relaxation_time # Legacy
        
        inv_T1s = 1/T1s
        
        kernel = np.sum(1 - 2*ps[:,None,None]*np.exp(-qs[:,None,None] * ts[None,:,None]/T1s[None,None,:]), axis=0) # K[i,j]
        kernel = np.matrix(kernel)
        
        kernel = np.diag(np.linalg.svd(kernel)[1])
        kernel = np.resize(kernel, (len(ts), len(T1s)))
        
        def gaussian(x, sigma):
            g = np.exp(-1/2 * np.square((T1s - x)/sigma))
            return g/np.maximum(1e-9, np.sum(g))
        
        def cost_function(M, K, P, alpha):
            P = np.abs(P)
            return np.square(np.linalg.norm(M - K@P)) + alpha*np.square(np.linalg.norm(P))
            
        bounds = [ [ 1e-9, 1.0] for i in range(num_bins) ]
        # WAY UNDERDETERMINED
        self.plotted_data = []
        for a in [1e-1, 1e0, 1e1]:
            #res = sp.optimize.differential_evolution(lambda x, *args: cost_function(args[0], args[1], x, a), bounds, args=(integrations, kernel), constraints=(sp.optimize.LinearConstraint(np.identity(num_bins), 1.0, 1.01),))
            #x0 = np.ones(num_bins)/num_bins#np.exp(-np.square(np.linspace(-10, 10, num_bins))/2)
            #res = sp.optimize.minimize(lambda x, *args: cost_function(args[0], args[1], x, a), x0=x0, args=(integrations, kernel), method='SLSQP', constraints=({'type': 'eq', 'fun': lambda x: 1-np.sum(np.abs(x))},), options={'ftol':1e-9, 'maxiter': 2500})
            x0 = sp.optimize.brute(lambda x, *args: cost_function(args[0], args[1], gaussian(x, 10), a), ranges=[(T1s[0], T1s[-1])], Ns=len(T1s), args=(integrations, kernel), finish=None)
            sigma = sp.optimize.brute(lambda x, *args: cost_function(args[0], args[1], gaussian(x0, np.exp(x)), a), ranges=[(1, np.log(T1s[-1]))], Ns=len(T1s), args=(integrations, kernel), finish=None)
            sigma = np.exp(sigma)
            print(x0, sigma)
            
            P0 = gaussian(x0, sigma)
            res = sp.optimize.minimize(lambda x, *args: cost_function(args[0], args[1], x, a), x0=P0, args=(integrations, kernel), method='SLSQP', constraints=({'type': 'eq', 'fun': lambda x: 1-np.sum(np.abs(x))},))
            print(res)
            
            res_x = np.abs(res.x)
            normed = res_x / np.sum(res_x)
            self.plotted_data += [(T1s, normed, f'alpha={a}')]
            self.plotted_data += [(T1s, P0, f'G alpha={a}')]
        
        self.update()
        