import numpy as np
import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.config import Config
from processor import Processor_Window
from plotter import Plotter_Window

kivy.require('2.0.0')

Config.set('kivy', 'exit_on_escape', '0')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

class WindowManager(ScreenManager):
    pass

class Processing(Processor_Window):
    pass

class Plotting(Plotter_Window):
    pass

class ZebrafishApp(App):
    def build(self):
        return wm_kv

if __name__ == '__main__':
    Builder.load_file('zebrafish.kv')
    Builder.load_file('processing.kv')
    Builder.load_file('plotting.kv')
    wm_kv = Builder.load_file('window_manager.kv')

    Window.size = (1200, 800)
    Window.top = 50
    Window.left = 100
    ZebrafishApp().run()
