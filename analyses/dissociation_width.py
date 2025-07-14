import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size':14, 'figure.autolayout': True})
from constants import BURGERS, ISF
from scipy.optimize import curve_fit


class Thermo():

    def __init__(self, path_to_mesh):
        mesh = np.load(path_to_mesh)
        widths_tmp = mesh.T[0].ravel() / BURGERS
        self.widths = widths_tmp[widths_tmp >= 0]
        self.stress = path_to_mesh.split('MPa')[0]
        self.temperature = path_to_mesh.split('/')[1].split('K')[0]

    def plot(self):
        #plt.hist(self.widths, bins=100, density=True, alpha=0.75, label=f"{self.stress} MPa | {self.temperature} K")
        bin_heights, bin_edges, _ = plt.hist(self.widths, bins=100, histtype='step', lw=2, alpha=0.75, density=True, label=f"{self.stress} MPa | {self.temperature} K")
        #plt.stairs(self.heights, self.edges, label=f"{self.stress} MPa | {self.temperature} K", fill=True, alpha=0.67)
        popt, x_fit, y_fit, half_max_x = fit_gaussian_to_peak(bin_edges, bin_heights)
        # Generate points for the fitted curve
        x_smooth = np.linspace(half_max_x, max(bin_edges), 1000)
        y_smooth = gaussian(x_smooth, *popt)
        plt.plot(x_smooth, y_smooth, '-k', lw=0.75, alpha=0.75)#, label=f'Fitted Gaussian {self.temperature}')
        print(f"Fitted parameters of {self.stress}/{self.temperature}: amplitude={popt[0]:.4f}, mean={popt[1]:.2f}, stddev={popt[2]:.2f}")


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)


def fit_gaussian_to_peak(bin_edges, bin_heights):
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find the peak
    max_y = np.max(bin_heights)
    peak_index = np.argmax(bin_heights)
    
    # Find the index where the curve first crosses half-max height to the left of the peak
    half_max_left_index = np.where((bin_heights[:peak_index] <= 0.5 * max_y) & 
                                   (bin_centers[:peak_index] < bin_centers[peak_index]))[0][-1]
    
    # Select data: all points to the right of half-max crossing
    mask = bin_centers >= bin_centers[half_max_left_index]
    x_fit = bin_centers[mask]
    y_fit = bin_heights[mask]
    
    # Initial guess for the fitting parameters
    amplitude = max_y
    mean = bin_centers[peak_index]
    stddev = np.std(x_fit - mean)
    
    # Perform the fit
    popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amplitude, mean, stddev])
    
    return popt, x_fit, y_fit, bin_centers[half_max_left_index]



paths = ['0MPa/350K/mesh_top_run5.npy',
         '0MPa/350K/mesh_top_run76.npy',
         '0MPa/500K/mesh_top_run105.npy',
         '0MPa/550K/mesh_top_run65.npy',
         '800MPa/100K/mesh_top_run10.npy',
         '800MPa/150K/mesh_bottom_run39.npy']

thermo_conditions = []
for path in paths:
    thermo_tmp = Thermo(path)
    thermo_tmp.plot()
    thermo_conditions.append(thermo_tmp)

plt.legend()
plt.xlabel('$d/b$')
plt.ylabel('Probability density')

plt.savefig('./dissociation_widths.pdf')
plt.show()