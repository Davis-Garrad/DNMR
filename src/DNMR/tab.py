import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *

import traceback

class Tab(QWidget):
    def __init__(self, data_widgets, name, parent=None):
        super(Tab, self).__init__(parent)
        
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.data_widgets = data_widgets
        self.data_widgets[name] = self
        
        self._name = name

        # layout stuff
        layout = QVBoxLayout()
        upper = self.generate_layout()
        if not(upper is None):
            layout.addLayout(upper)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.fig.add_subplot(111)

        self.fileselector = data_widgets['fileselector'] # Keep at bottom - cannot be used until file is read!
        self.fileselector.callbacks += [self.update]

    def generate_layout(self):
        print(f'GENERATE_LAYOUT ({self._name})')
        return None
    
    def update(self):
        print(f'UPDATE ({self._name})')
        self.plot()

    def plot_logic(self):
        print(f'UNIMPLEMENTED PLOT_LOGIC ({self._name})')
        pass

    def plot(self):
        if(self.fileselector.fn == ''):
            return
        # save the current zoom for restoring later
        old_x_lim = self.ax.get_xlim()
        old_y_lim = self.ax.get_ylim()
        
        if not(self.fileselector.checkbox_holdplots.isChecked()):
            self.ax.clear()

        try:
            print(f'PLOT_LOGIC ({self._name})')
            self.plot_logic()
        except:
            print(f'Failure in plot_logic\n{"-"*100}')
            traceback.print_exc()
            print("-"*100)
            
        # Thanks, azelcer (https://stackoverflow.com/questions/70336467/keep-zoom-and-ability-to-zoom-out-to-current-data-extent-in-matplotlib-pyplot)
        self.ax.relim()
        self.ax.autoscale()
        self.toolbar.update() # Clear the axes stack
        self.toolbar.push_current()  # save the current status as home

        self.ax.set_xlim(old_x_lim)  # and restore zoom
        self.ax.set_ylim(old_y_lim)
        self.ax.legend()

        self.canvas.draw()