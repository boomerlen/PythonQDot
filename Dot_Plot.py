# Dot_Plot.py
#
# A class and some functions that simplify some common plotting techniques for quantum dots
#
# Hugo Sebesta, 2021

import matplotplib.pyplot as plt

class Dot_Plot:
    'Class container for common plotting techniques with matplotlib'

    def __init__(self, no_points):
        self.x_axis = np.arange(0, no_points)

        self.fig, self.axis = plt.subplots()

    def add_plot(self, quantity):
        self.axis.plot(self.x_axis, quantity)

    def set_title(self, _title):
        self.axis.set(title=_title)

    def display(self):
        plt.show()

def plot(quantity):
    'Plots the given quantity in the simplest way imaginable'
    no_points = len(quantity)

    dot_plot = Dot_Plot(no_points)

    dot_plot.add_plot(quantity)

    dot_plot.display()

    return
