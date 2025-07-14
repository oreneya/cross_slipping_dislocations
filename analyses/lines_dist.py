import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size':15, 'figure.autolayout': True})
from scipy import ndimage
from glob import glob
import numpy as np
from scipy.optimize import curve_fit
from constants import BURGERS
import re
from scipy.stats import gamma
import sys, os


def read_simulation_file(file):
    runid = file.split('.')[0].split('run')[1]
    with open(file, 'r') as f:
        data = f.readlines()
    return data, runid


def process(data, time_coord):
    LC = []
    for row in data[:time_coord]:
        lines = re.split('\[|\]|, |\n', row)[1:-2]
        if len(lines[0]) > 0:
            [LC.append(float(line)) for line in lines]
    return LC


def model(data):
    """Implementation of the probabilistic model"""
    # get P(L) data
    hist = np.histogram(data, bins=200, density=True)
    P = hist[0]
    L = hist[1][:-1]
    
    # fit model
    def func(x, p):
        return x * np.power(p, x-1) * np.power(1-p, 2)
    
    popt, pcov = curve_fit(func, L, P, p0=0.3)
    x_fit = np.arange(0, 1.1*L[-1], L[-1]/400)
    y_fit = func(x_fit, popt[0])

    return x_fit, y_fit, popt[0], L, P


def model2(data):
    """Same model but with a term accounting for an elastic interaction between the 2-half PCs."""
    # get P(L) data
    hist = np.histogram(data, bins=200, density=True)
    P = hist[0]
    L = hist[1][:-1]
    
    # fit model
    def func(x, p, gamma, A, alpha):
        return A * x * np.power(p, x-1) * np.power(1-p, 2) * (1 / (1 + np.power(x, -alpha) / gamma))
    
    popt, pcov = curve_fit(func, L, P, p0=[0.5, 0.125, 4, 5])
    print(popt)
    x_fit = np.arange(0, 1.1*L[-1], L[-1]/400)
    y_fit = func(x_fit, *popt)
    
    return x_fit, y_fit, popt[0], popt[1], popt[2], popt[3], L, P


def plot(results, x, y, p, context):
    fig, ax = plt.subplots()
    ax.hist(results, bins=200, density=True)
    ax.plot(x, y, '--k')
    
    ax.set_xlabel('$L\ (b)$')
    ax.set_ylabel('Probability density')
    # add text box for the statistics
    stats = (f"$T$ = {context['temperature']}\n"
             f"$p$ = {p:.2f}\n"
             f"$mode$ = {x[np.argmax(y)]:.2f}")
    bbox = dict(boxstyle='round', ec='black', alpha=0.25)
    ax.text(0.95, 0.07, stats, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')
    plt.savefig(f"{PATH_RESULTS}dist_all_LC_{context['material']}_{context['stress']}_{context['temperature']}.pdf")
    #plt.close()
    plt.show()


def plot2(results, x, y, p, gamma, A, alpha, context):
    fig, ax = plt.subplots()
    ax.hist(results - 0, bins=200, density=True)
    ax.plot(x, y, '--k')
        
    ax.set_xlabel('$L\ (b)$')
    ax.set_ylabel('Probability density')
    # add text box for the statistics
    stats = (f"$T$ = {context['temperature']}\n"
             f"$p$ = {p:.2f}\n"
             f"$\gamma$ = {gamma:.2f}\n"
             f"$\\alpha$ = {alpha:.2f}\n"
             f"$A$ = {A:.2f}\n"
             f"$mode$ = {x[np.argmax(y)]:.2f}")
    bbox = dict(boxstyle='round', ec='black', alpha=0.25)
    ax.text(0.95, 0.07, stats, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')
    plt.savefig(f"{PATH_RESULTS}dist_all_LC_{context['material']}_{context['stress']}_{context['temperature']}.pdf")
    #plt.close()
    plt.show()


def main():

    # set path
    global PATH_RESULTS
    PATH_RESULTS = sys.argv[1]
    #PATH_RESULTS = '0MPa/500K/'

    # context-conditions
    context = {}
    context['material'] = 'aluminum'
    stress, temperature = PATH_RESULTS.split('/')[:2]
    context['stress'] = stress
    context['temperature'] = temperature

    files = glob(f"{PATH_RESULTS}lines*csv")
    results = []

    for simulation_file in files:
        data, runid = read_simulation_file(simulation_file)
        # if the file (top/bottom) has a full CS event in it, extract its initiation timing, otherwise it's time=0
        try:
            nuc_mat = np.load(f"{PATH_RESULTS}nucleation_matrix_run_{runid}.npy")
            time_coord = nuc_mat.nonzero()[1].min() # first non-zero pixel in time axis of nucleation matrix
        except:
            time_coord = 0
        results.extend(process(data, time_coord))
    
    # normalize to Burgers
    results = np.array(results) / BURGERS

    # fit & plot
    x, y, p, L, P = model(results)
    plot(results, x, y, p, context)

    # fit & plot with elastic interaction
    x, y, p2, gamma, A, alpha, L, P = model2(results)
    plot2(results, x, y, p2, gamma, A, alpha, context)

    # output p and mode
    with open(f'{PATH_RESULTS}PvL_mode', 'w') as f:
        f.writelines(str(x[np.argmax(y)]))
    with open(f'{PATH_RESULTS}PvL_avg', 'w') as f:
        f.writelines(str(np.average(results)))
    with open(f'{PATH_RESULTS}PvL_raw_mode', 'w') as f:
        f.writelines(str(L[np.argmax(P)]))
    # fitting parameters
    with open(f'{PATH_RESULTS}PvL_p', 'w') as f:
        f.writelines(str(p2))
    with open(f'{PATH_RESULTS}PvL_A', 'w') as f:
        f.writelines(str(A))
    with open(f'{PATH_RESULTS}PvL_gamma', 'w') as f:
        f.writelines(str(gamma))
    with open(f'{PATH_RESULTS}PvL_p_without_interaction', 'w') as f:
        f.writelines(str(p))

if __name__ == '__main__':
    main()