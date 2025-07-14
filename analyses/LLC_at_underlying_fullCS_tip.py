import numpy as np
from scipy import ndimage
from glob import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size':15, 'figure.autolayout': True})
from matplotlib.ticker import MaxNLocator


def process_mesh(fn):

    mesh = np.load(fn)

    # get run id & top/bottom
    top_or_bot = fn.split('_')[-2]
    if top_or_bot == 'bottom':
        top_or_bot = 'bot'
    run_id = fn.split('run')[-1].split('.')[0]

    # get which dislocation has the full CS (top or bottom)
    attempts_path_fullCS = glob(f"{path}attempts_run{run_id}_{top_or_bot}_fullCS")

    # obtain nucleation timings for a cutoff
    if attempts_path_fullCS:
        nuc_mat = np.load(f"{path}nucleation_matrix_run_{run_id}.npy")
        time_cutoff = nuc_mat.nonzero()[1].min() # use max (min) for end (start) time coord of nucleation island
    else:
        return
            
    # extract CS & its LC wrapper segments
    mesh_comb = mesh.T[2].T + mesh.T[1].T
    labels_comb, _ = ndimage.label(mesh_comb)
    labels_comb_count = np.bincount(labels_comb.ravel())

    CS_lengths = np.array([])
    comb_label_lengths = np.array([])

    # clean mesh from LC & CS anywhere except those in the fullCS island
    fullCS_Lcomb_label = np.argmax(labels_comb_count[1:]) + 1
    fullCS_matrix = labels_comb == fullCS_Lcomb_label
    mesh.T[1] *= fullCS_matrix.T
    mesh.T[2] *= fullCS_matrix.T
    
    return mesh, run_id


def find_shape_edges(mesh):

    # define matrix as the combination of LC and CS
    comb_matrix = mesh.T[1] + mesh.T[2]
    
    # Find the first and last occurrence of 1 in each column
    top_edge = np.argmax(comb_matrix, axis=0)
    bottom_edge = comb_matrix.shape[0] - 1 - np.argmax(comb_matrix[::-1, :], axis=0)
    
    # Handle columns with no 1s
    no_shape = np.all(comb_matrix == 0, axis=0)
    top_edge[no_shape] = -1
    bottom_edge[no_shape] = -1
    
    return top_edge, bottom_edge


def linear_fit(points):
    """
    Fit a line to the time window of CS growth stage.
    Split the data into 10 segments with varying lengths by trimming
    in the direction of the time flow. Rank each trimmed case by
    goodness of fit. Return the longest trimmed that its goodness
    of fit is one of the majority.
    """
    
    n = 10 # number of trims
    x = np.where(points != -1)[0]
    y = points[x]
    A = np.vstack([x, np.ones(len(x))]).T
    m = np.empty(n)
    c = np.empty(n)
    
    for i in range(n):
        Ai = A[i//len(x):]
        yi = y[i//len(x):]
        mi, ci = np.linalg.lstsq(Ai, yi, rcond=None)[0]
        m[i] = mi
        c[i] = ci
    
    i_optimal = np.argmin(np.gradient(m))

    return m[i_optimal], c[i_optimal]


def find_intersection(line1, line2):
    """Find the intersection point of two lines."""
    m1, b1 = line1
    m2, b2 = line2
    
    if m1 == m2:
        return None  # Lines are parallel
    
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    
    return int(x), int(y)


def plot_diagnostics(mesh, intersection, top_slope, bottom_slope, context, run_id):
    """
    Plot mesh of full CS island, layered with the extrapolation lines
    of the CS growth stage, and their intersection.
    """

    time, position = intersection
    fig, ax = plt.subplots()
    ax.imshow(mesh.T[1] + 2*mesh.T[2])
    ax.axvline(time, alpha=0.75)
    ax.axline((time, position), slope=top_slope, alpha=0.75)
    ax.axline((time, position), slope=bottom_slope, alpha=0.75)
    
    new_ticks = (ax.get_xticks() / 100).astype(int)
    ax.set_xticklabels(new_ticks)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel("Whole dislocation (b)")
    # plot for time axis starting just before the tip
    time2 = time - int(0.1*(mesh.shape[0] - time))
    time2 = max(time2, 0)
    plt.xlim(left=time2)
    plt.savefig(f"{path}fullCS_extrapolated_tip_{context['material']}_{context['stress']}_{context['temperature']}_run{run_id}.png", dpi=300)
    plt.show()


def main():

    # set path to working folder
    global path
    path = "800MPa/100K/"

    # context-conditions
    context = {}
    context['material'] = 'aluminum'
    stress, temperature = path.split('/')[:2]
    context['stress'] = stress
    context['temperature'] = temperature

    # process batch of simulations
    files = glob(path + "mesh*.npy")
    LCS_collection = []
    Lcomb_collection = []
    for fn in files:
        print(fn)
        # get the full CS nucleation + growth island
        try:
            mesh, run_id = process_mesh(fn)
        except:
            print(f"Probably {fn} is missing attempts_runxxx file")
            continue
        # isolate the top & bottom edges of the triangle-like shape
        top_edge, bottom_edge = find_shape_edges(mesh)
        # obtain the line formulas of the growth phase in this 2d space
        top_line = linear_fit(top_edge)
        bottom_line = linear_fit(bottom_edge)
        # calculate the virtual intersection of these growth lines
        intersection = find_intersection(top_line, bottom_line)
        inter_time = intersection[0]
        # collect LLC and LCS at intersection time
        LLC = sum(mesh.T[1].T[inter_time])
        LCS = sum(mesh.T[2].T[inter_time])
        Lcomb = LLC + LCS
        LCS_collection.append(LCS)
        Lcomb_collection.append(Lcomb)
        plot_diagnostics(mesh, intersection, top_line[0], bottom_line[0], context, run_id)

    # plot
    plt.hist(Lcomb_collection)
    plt.xlabel("LC + CS length (b)")
    plt.savefig(f"{path}fullCS_tip_{context['material']}_{context['stress']}_{context['temperature']}.pdf")


if __name__ == "__main__":
    main()