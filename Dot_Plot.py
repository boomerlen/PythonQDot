# Dot_Plot.py
#
# A class and some functions that simplify some common plotting techniques for quantum dots
#
# Hugo Sebesta, 2021

import numpy as np 
import matplotlib.pyplot as plt

def simple_plot(quantity):
    'Plots the given quantity in the simplest way imaginable'
    no_points = len(quantity)

    fig, axis = plt.subplots()

    x_axis = np.arange(no_points)

    axis.plot(x_axis, quantity)

    plt.show()

    return
