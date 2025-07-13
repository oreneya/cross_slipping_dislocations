"""Algorithm to obtain from dislocation-extraction-algorithm (DXA of A. Stukowski
as implemented in OVITO) features occurring along a dissociated dislocation
during a simulation of screw dislocations in a dipole configuration of face-
centeredc cubic metals. Those features include dissociation width, line
constrictions, and occurrances of cross-slip via either Friedel-Escaig or
Fleischer mechanisms. Extracted segments are saved in a mesh-grid with one
dimension as time and another as the dislocations sense-line."""

# imports (requires ovitos as python interpreter)
from glob import glob
import os
import numpy as np
from ovito.io import import_file
import sys

# --- GLOBAL SETTINGS --- #
# paths
PATH_DATA = sys.argv[1]
PATH_RESULTS = PATH_DATA + '/results'
if not os.path.isdir(PATH_RESULTS):
    os.mkdir(PATH_RESULTS)
PATH_LOGS = PATH_DATA + '/logs'
if not os.path.isdir(PATH_LOGS):
    os.mkdir(PATH_LOGS)

# how many pixels should the mesh be in the dislocation-length dimension
PIXELS = 200

# how many layers are computed for imshow (dissociation widths + dislocation types)
LAYERS = 4

# reciprocal of Shockley partials Burgers vector
RSB = 6

# --- END GLOBAL --- #

class DislocationReconstruction():
    """A class to represent each half of a dissociated dislocation, 4 in total."""

    def __init__(self):
        # DXA node points that comprise the quadrant
        self.xy_right = np.array([[], []])
        self.xy_left = np.array([[], []])
        self.lc = []
        self.fe = []
        self.fl = []
        self.z = np.array([])

    def extend_xy(self, z, xy, right, seg_type):
        """Extend the lists of segments with new incoming x-y pairs,
        and do it in parallel for all 4 channels of segment types."""

        if seg_type['shockley'] and not seg_type['xslip']:
            if right:
                self.xy_right = np.append(self.xy_right, xy, axis=1)
            else:
                self.xy_left = np.append(self.xy_left, xy, axis=1)
        elif seg_type['lc']:
            self.lc.append([xy[1][0], xy[1][-1]])
        elif seg_type['shockley'] and seg_type['xslip'] and right:
            # taking only right b/c projected length of the dissociated
            # dislocation is identical for both right & left Shockley
            # partials.
            self.fe.append([xy[1][0], xy[1][-1]])
        elif seg_type['stairrod']:
            self.fl.append([xy[1][0], xy[1][-1]])
        else: pass

        # saving z for later use in rebuild_dislocations
        self.z = np.append(self.z, z)

    def rebuild_dislocations(self, matrix):
        """Redistribute the dislocation nodes into a fixed-size array of floats,
        representing the whole dislocation, ready for width calculation against
        its counterpart dislocation."""

        # end analysis at an advanced stage of the annihilation process
        if self.xy_left.shape[1] == 0 or self.xy_right.shape[1] == 0:
            return 0

        # --- get boundary markers for new mesh-arrays --- #

        # get relevant simulation cell geometry values
        yz_tilt = matrix[1][2]
        lz = matrix[2][2]
        ly = matrix[1][1]
        origin_y = matrix[1][3]

        # Define mesh-array boundaries.
        # Mean for z-values of dislocations, assuming origin of z-axis is
        # exactly between the top & bottom dislocations.
        z = np.mean(self.z)
        y0 = origin_y + yz_tilt * (lz/2 + z) / lz
        y1 = y0 + ly

        #  --- construct & populate mesh-arrays --- #

        # construct
        mesh_vals = np.linspace(y0, y1, PIXELS)
        mesh = np.zeros((LAYERS, PIXELS)) # 'ovitos... if it only knew what pandas is :/ '

        ####################
        #     R&L only     #
        ####################

        # interploate into mesh either side of 'dislocation' line
        mesh_xl = np.interp(mesh_vals, self.xy_left[1], self.xy_left[0], period=ly)
        mesh_xr = np.interp(mesh_vals, self.xy_right[1], self.xy_right[0], period=ly)

        # first mesh as dissociation width
        mesh[0] = mesh_xr - mesh_xl

        ####################
        #     LC/FE/FL     #
        ####################

        # sort by y-axis
        self.lc.sort()
        self.fe.sort()
        self.fl.sort()

        # match idx & populate mesh
        for i, seg_type in enumerate([self.lc, self.fe, self.fl]):
            mesh[i+1] = match_and_populate(seg_type, mesh_vals, mesh[i+1], ly)

        return mesh.T

def match_and_populate(seg_type, mesh_values, meshi, ly):
    """Populating with ones the LC/FE/FL segment types"""

    # initialize fake indices
    idx0 = idx1 = 0

    for edges in seg_type:

        # both inside
        if edges[0] >= mesh_values[0] and edges[1] <= mesh_values[-1]:
            idx0 = np.argmin(abs(mesh_values - edges[0])).astype(int)
            idx1 = np.argmin(abs(mesh_values - edges[1])).astype(int)

        # first outside
        elif edges[0] < mesh_values[0] and edges[1] >= mesh_values[0]:
            shifted_edge = edges[0] + ly
            idx_tmp = np.argmin(abs(mesh_values - shifted_edge)).astype(int)
            idx0 = idx_tmp - len(mesh_values)
            idx1 = np.argmin(abs(mesh_values - edges[1])).astype(int)

        # second outside
        elif edges[0] <= mesh_values[-1] and edges[1] > mesh_values[-1]:
            shifted_edge = edges[1] - ly
            idx_tmp = np.argmin(abs(mesh_values - edges[0])).astype(int)
            idx0 = idx_tmp - len(mesh_values)
            idx1 = np.argmin(abs(mesh_values - shifted_edge)).astype(int)

        # no option for both outside in either direction so ending switches here
        else:
            pass

        # in case there are instances of special segments
        if idx0 != idx1:
            # vectorize indices
            idx = range(idx0, idx1)
            meshi[idx] = 1

    return meshi

def analyze_timestep(filepath):
    """Collect & analyze data at a given timestep stored in ovito crystal-
    analysis (*.ca) files via ovitos api"""

    # load DXA data
    pipeline = import_file(filepath)
    data = pipeline.source

    # initialize four 'dislocations' lists of objects to then rebuild and use
    # to calc. diss. width
    top_dis = DislocationReconstruction()
    bottom_dis = DislocationReconstruction()

    for segment in data.dislocations.segments:

        # extract top vs bottom
        z = segment.points.T[2]

        # end analysis if advanced CS
        if min(abs(z)) < 0.1 * max(z):
            return 0, 0

        # determine dislocation's segment type
        seg_type = segment_type(segment)

        # extract right vs left
        right = segment.spatial_burgers_vector[0] > 0

        # x-y pairs defining nodes on the segment
        xy = segment.points.T[:2]

        if z[0] > 0:
            top_dis.extend_xy(z, xy, right, seg_type)
        else:
            bottom_dis.extend_xy(z, xy, right, seg_type)

    # throw a warning in case collected data is denser than the predefined resolution
    max_cum_nodes = max(cum_nodes(top_dis), cum_nodes(bottom_dis))

    if max_cum_nodes > PIXELS:
        resolution_warning = "Detected cumulative number of nodes {},\
        but PIXELS is set to only {}, filepath: {}".format(max_cum_nodes, PIXELS, filepath)
        with open(PATH_LOGS + '/warnings', 'a') as f:
            f.writelines(resolution_warning)

    # convert Cartesian coordinates into meshgrid to plot with imshow
    res_top = top_dis.rebuild_dislocations(data.cell.matrix)
    res_bottom = bottom_dis.rebuild_dislocations(data.cell.matrix)

    return res_top, res_bottom

def cum_nodes(dis):
    """obtain cumulative nodes per dislocation line to compare with PIXELS"""

    len_xy = max(len(dis.xy_right), len(dis.xy_left))
    len_rest = len(dis.lc) + len(dis.fe) + len(dis.fl)
    return len_xy + len_rest

def segment_type(segment: object)->dict:
    """Determine type of dislocation segment."""

    # transform segment representation to usable format
    seg = np.rint(RSB * np.array(segment.true_burgers_vector))

    # remove assumptions
    seg_type = {'shockley': False,
                'lc': False,
                'stairrod': False,
                'xslip': False}

    # check if segment is a perfect dislocation (LC)
    if sum(abs(seg)) == RSB and np.prod(seg) == 0:
        seg_type['lc'] = True

    # check if segment is a Shockley partial
    if abs(np.prod(seg)) == 2:
        seg_type['shockley'] = True

    # segment is neither LC nor Shockley partial => stair-rod
    if not seg_type['lc'] and not seg_type['shockley']:
        seg_type['stairrod'] = True

    # check if segment is on the x-slip plane
    if seg_type['shockley'] and abs(segment.spatial_burgers_vector[2]) > 0.5:
        seg_type['xslip'] = True

    return seg_type

def list_files(run_num):
    """Prepare for analysis list of files as a time-series."""

    flist = []
    file_list = glob("{}/*run{}.*.ca.gz".format(PATH_DATA, run_num))

    for filepath in file_list:
        zfilled = filepath.split('.')[-3].zfill(6)
        flist.append([zfilled, filepath])
    flist.sort()
    return flist

def main():
    """Prepare files for consumption, then iteratively analyze_timesteps, and
    save accumulated results for later visualization and further analysis"""

    # prepare input
    run_num = sys.argv[2]
    batch = sys.argv[3]
    flist = list_files(run_num)

    # initialize arrays for imshow (4 layers of data for dislocations state)
    mesh_top = np.empty((len(flist), PIXELS, LAYERS))
    mesh_bottom = np.empty((len(flist), PIXELS, LAYERS))

    # now iterate through simulation's files
    for i, filepath in enumerate(flist):

        # rename path
        filepath = filepath[1]

        # feed execution monitoring file
        with open("{}/exec_status_{}".format(PATH_LOGS, run_num), 'w') as f:
            f.writelines("file {}/{}, filepath {}".format(i, len(flist), filepath))

        # analyze current snapshot
        tmp_top, tmp_bot = analyze_timestep(filepath)

        # either end analysis due to annihilation or keep current timestep results
        if isinstance(tmp_top, np.ndarray) and isinstance(tmp_bot, np.ndarray):
            mesh_top[i], mesh_bottom[i] = tmp_top, tmp_bot
        else:
            break

    # save results to disk
    np.save("{}/mesh_top_run{}_{}.npy".format(PATH_RESULTS, run_num, batch), mesh_top[:i])
    np.save("{}/mesh_bottom_run{}_{}.npy".format(PATH_RESULTS, run_num, batch), mesh_bottom[:i])

if __name__ == "__main__":
    main()
