
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

class TabT1Fit(Tab):
    output_frames = {}

    def __init__(self, data_widgets, parent=None):
        super(TabT1Fit, self).__init__(data_widgets, 'tab_t1_fitting', parent)
        
        self.data = (np.array([]), np.array([]))
        self.plot_data = (np.array([]), np.array([]))
        self.excluded_points_indices = []
        self.x0 = None
        self.sigmas = None

    def generate_layout(self):
        self.combobox_fittingroutine = QComboBox()
        self.combobox_fittingroutine.currentIndexChanged.connect(self.update_fit_type)
        
        self.pushbutton_fit = QPushButton('Fit')
        self.pushbutton_fit.clicked.connect(self.fit)
        
        self.checkbox_normalize = QCheckBox('Normalize?')

        l = QHBoxLayout()
        lv = QVBoxLayout()
        lv.addWidget(self.combobox_fittingroutine)
        lv.addWidget(self.checkbox_normalize)
        l.addLayout(lv)
        l.addWidget(self.pushbutton_fit)

        def add_fit_frame(name, *args):
            # fit output
            frm = QFrame() # TODO: Make this better.
            frm.hide()
            lo = QVBoxLayout()
            self.output_frames[name] = [ frm ]
            for i in range(len(args)//2):
                w = FitParameterWidget(args[2*i], args[2*i+1])
                #w = QLineEdit('fitting...')
                #fix = QCheckBox('Fix?')
                #li = QHBoxLayout()
                #li.addWidget(fix)
                #li.addWidget(w)
                lo.addWidget(w)
                self.output_frames[name] += [ {'widget': w } ]
            self.combobox_fittingroutine.addItem(name)
                
            frm.setLayout(lo)
            l.addWidget(frm)

        # Title, var_name, var_units, var_name, var_units, ...
        add_fit_frame('7/2 Spin', '\u03b30', '', 's', '', 'T1', '\u03bcs', 'r', '')
        add_fit_frame('7/2 Spin (Sat. 1)', '\u03b30', '', 's', '', 'T1', '\u03bcs', 'r', '')
        #add_fit_frame('1/2 Spin', '\u03b30', '', 's', '', 'T1', '\u03bcs', 'r', '')
        #add_fit_frame('Spin 1', '\u03b30', '', 's', '', 'T1', '\u03bcs', 'r', '')
        # ...
        
        self.update_fit_type()
        
        self.canvas.mpl_connect('button_press_event', self.process_button)

        return l

    def process_button(self, event):
        if(event.button == 1):
            if not(event.xdata is None):
                screenspace_data = self.ax.transData.transform(np.array([self.data[0], self.data[1]]).T).T
                screenspace_click = self.ax.transData.transform((event.xdata, event.ydata))
                
                xdist = np.square(screenspace_click[0] - screenspace_data[0])
                ydist = np.square(screenspace_click[1] - screenspace_data[1])
                selected_point_index = np.argmin(xdist + ydist)
                if(selected_point_index in self.excluded_points_indices):
                    self.excluded_points_indices.remove(selected_point_index)
                else:
                    self.excluded_points_indices += [selected_point_index]
                self.update()

    def plot_logic(self):
        freq = self.data_widgets['tab_ft'].data[0]
        ft   = self.data_widgets['tab_ft'].data[1]
        real = np.real(ft)
        try:
            del_times = self.fileselector.data.sequence['0'].delay_time
        except:
            del_times = self.fileselector.data.sequence['0'].relaxation_time # Legacy
            print(del_times)

        integrations = np.zeros(real.shape[0], dtype=np.complex128)
        start_index = np.argmin(np.abs(self.data_widgets['tab_ft'].left_pivot - freq))
        end_index = np.argmin(np.abs(self.data_widgets['tab_ft'].right_pivot - freq))
        if(end_index < start_index):
            tmp = start_index
            start_index = end_index
            end_index = tmp

        integrations = np.sum(real[:,start_index:end_index], axis=1)
        
        if(self.checkbox_normalize.isChecked()):
            integrations /= np.max(integrations)
        rt = np.real(self.data_widgets['tab_ft'].data[1])
        
        uncertainties = 1e-6*np.ones_like(integrations) # TODO: Figure out real stddevs np.std(np.append(rt[:,:start_index], rt[:,end_index:], axis=1), axis=1)*np.sqrt(end_index-start_index)*np.ones_like(integrations) # TODO: Figure out real stddevs
        #uncertainties += integrations * np.sqrt((end_index-start_index+1) / rt.shape[1])
        #uncertainties = np.abs(uncertainties)
            
        sort_indices = np.argsort(del_times)
        del_times = del_times[sort_indices]
        integrations = integrations[sort_indices]
        uncertainties = uncertainties[sort_indices]
        
        self.ax.set_xscale('log')
        self.ax.set_xlabel('delay time (us)')
        plotted_integrations = []
        plotted_del_times = []
        plotted_errs = []
        for i in range(len(integrations)):
            if not(i in self.excluded_points_indices):
                plotted_integrations += [integrations[i]]
                plotted_del_times += [del_times[i]]
                plotted_errs += [uncertainties[i]]
        plt_pts = self.ax.errorbar(plotted_del_times, plotted_integrations, label='integrations', linestyle='', marker='o', yerr=plotted_errs)
        self.ax.scatter(del_times[self.excluded_points_indices], integrations[self.excluded_points_indices], color=(plt_pts[-1][-1]).get_color(), linestyle='', marker='x')
        
        post_aq_max = np.max(self.fileselector.data.params.post_acquisition_time * 1e3) # this is in ms. Our axes in us
        self.ax.axvline(post_aq_max, linestyle='--', color='k')

        self.data = (del_times, integrations, uncertainties)

        if(self.plot_data[0].shape[0] > 0):
            params_list = ''
            out_frame = self.output_frames[self.combobox_fittingroutine.currentText()]
            for i in out_frame[1:]:
                params_list += f'{i['widget'].get_full_display()}\n'
            params_list = params_list[:-1]
            self.ax.plot(self.plot_data[0], self.plot_data[1], label=params_list)
        
    def update_fit_type(self):
        for key, val in self.output_frames.items():
            val[0].hide()
        out_frame = self.output_frames[self.combobox_fittingroutine.currentText()]
        out_frame[0].show()
        
    def fit(self):
        self.update() # get most recent values to fit
        self.plot_data = (np.array([]),np.array([]))
        out_frame = self.output_frames[self.combobox_fittingroutine.currentText()]
        bounds = None
        try:
            del_times = self.fileselector.data.sequence['0'].delay_time
        except:
            del_times = self.fileselector.data.sequence['0'].relaxation_time # Legacy, as I didn't know what this was when I wrote it. Surprise, surprise

        if(self.combobox_fittingroutine.currentText() == '7/2 Spin'):
            # DEVELOPER NOTE: If you want to add more options for this, make sure to define fit_func (similarly to below) and add an item in the generate_layout function
            
            bounds = [ [0, np.max(np.abs(self.data[1]))*10], [-1, 10], [np.min(del_times)/10, np.max(del_times)*10], [0.99*0, 1.01*10] ]
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
                
        elif(self.combobox_fittingroutine.currentText() == '7/2 Spin (Sat. 1)'):
            bounds = [ [0, np.max(np.abs(self.data[1]))*10], [-1, 10], [np.min(del_times)/10, np.max(del_times)*10], [0.99*0, 1.01*10] ]
            
            def fit_func(args, t):
                gamma_0 = args[0]
                s = args[1]
                T1 = args[2]
                r = args[3]
                
                return gamma_0 * (1 - (1+s) * (1/84*np.exp(-np.pow(t/T1, r)) + 
                                               1/84*np.exp(-np.pow(3*t/T1, r)) + 
                                               2/66*np.exp(-np.pow(6*t/T1, r)) + 
                                               18/154*np.exp(-np.pow(10*t/T1, r)) + 
                                               1/1092*np.exp(-np.pow(15*t/T1, r)) + 
                                               49/132*np.exp(-np.pow(21*t/T1, r)) + 
                                               392/858*np.exp(-np.pow(28*t/T1, r))))
            
            
            
        def cost_func(args, x, y, yerr):
            return np.sum(np.square((fit_func(args, x) - y)/np.maximum(yerr, 0.01))) # more points is more fits
            
        for i in range(len(out_frame)-1):
            if(out_frame[i+1]['widget'].is_fixed()):
                # Fix
                fv = out_frame[i+1]['widget'].get_value()
                bounds[i] = [ fv, fv ]
            
        included_xvals = []
        included_yvals = []
        included_errs = []
        for i in range(len(self.data[0])):
            if not(i in self.excluded_points_indices):
                included_xvals += [self.data[0][i]]
                included_yvals += [self.data[1][i]]
                included_errs  += [self.data[2][i]]
        included_xvals = np.array(included_xvals)
        included_yvals = np.array(included_yvals)
        included_errs  = np.array(included_errs)
        # global minimum
        res = sp.optimize.differential_evolution(lambda x: cost_func(x, 
                                                                     included_xvals, 
                                                                     included_yvals,
                                                                     included_errs), 
                                                 bounds=bounds)
        # get uncertainties on the fit, as I am too lazy to do the full analysis when scipy will do it for me
        try:
            picky_scipy_bounds = np.array(bounds).T 
            picky_scipy_bounds[0,:] -= 1e-9
            popt, pcov = sp.optimize.curve_fit(lambda xs, *args: fit_func(args, xs), included_xvals, included_yvals, p0=res.x, bounds=picky_scipy_bounds, sigma=included_errs, absolute_sigma=False)
        
            print(res)
            self.x0 = popt
            self.sigmas = np.sqrt(np.diag(pcov))
            x_vals = included_xvals
            if(x_vals.shape[0] < 100):
                x_vals = np.exp(np.linspace(np.log(np.min(x_vals*1e-1)), np.log(np.max(x_vals*1e1)), 100, endpoint=True))
            self.plot_data = (x_vals, fit_func(popt, x_vals))
            for i in range(len(self.x0)):
                try:
                    digits = int(np.ceil(np.abs(np.log10(self.sigmas[i]))))
                except:
                    digits = 10000 # sigma negative - a sign that something has gone horribly wrong and the user should deal with the drama. Show them the digits.
                if(self.sigmas[i] > 1.0):
                    rounded_digits = -digits+1
                else:
                    rounded_digits = digits
                    
                display_sigma = np.round(self.sigmas[i], rounded_digits)
                display_x = np.round(self.x0[i], rounded_digits)
                
                if not(out_frame[i+1]['widget'].is_fixed()):
                    out_frame[i+1]['widget'].set_value(display_x, display_sigma)
                #out_frame[i+1]['widget'].setText(f'{out_frame[i+1]["label"]}={display_x:,}\u00b1{display_sigma:,}{out_frame[i+1]["units"]}')
        except Exception as e:
            traceback.print_exc()
        self.update()
        
    def get_exported_data(self):
        out_frame = self.output_frames[self.combobox_fittingroutine.currentText()][1:]
        params_dict = {}
        if(self.x0 is not None):
            cnt = 0
            for i in out_frame:
                params_dict[i['label']] = [ str(self.x0[cnt]) + ' ' + i.units ]
                params_dict[i['label']+' error'] = [ str(self.sigmas[cnt]) + ' ' + i.units ]
                cnt += 1
        
        index = self.fileselector.spinbox_index.value()
        pd = {
                 'frequencies (MHz)': self.data_widgets['tab_ft'].data[0],
                 'fft': self.data_widgets['tab_ft'].data[1][index],
                 'delays': self.data[0],
                 'integrals': self.data[1],
                }
        pd.update(params_dict)
        return pd