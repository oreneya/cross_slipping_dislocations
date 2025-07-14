from scipy import ndimage
import numpy as np

class Nucleation:
    """Class to characterize a nuleation event of a full CS."""

    def __init__(self, data_top, data_bottom):
        """Initialize with top & bottom dislocation mesh matrices"""

        self.data_top = data_top
        self.data_bottom = data_bottom
        self.top_or_bottom = None

    def location(self):
        """
        Process the binary matrix of FE CS (for both top & bottom dislocations),
        and return the location of the initial moment of nucleation on the
        spacetime map (and whether it is associated with the top or bottom
        dislocations).
        """

        top = locate(self.data_top.T[2], self.data_top.T[1])
        bot = locate(self.data_bottom.T[2], self.data_bottom.T[1])

        if top[0] == bot[0] == 0:
            return

        if top[0] > bot[0]:
            out = top
            self.top_or_bottom = 'top'
        else:
            out = bot
            self.top_or_bottom = 'bottom'

        self.time = out[1]
        self.segment = [out[2][0], out[2][-1]]
        self.region = out[3]


def detect(data_cs):
    """Detect a nucleation event."""

    # characterize full cs 'triangle'
    labels_cs, _ = ndimage.label(data_cs, np.ones((3, 3)))
    labels_count_cs = np.bincount(labels_cs.ravel())
        
    # not even 1 attempt is present, exit
    if len(labels_count_cs) == 1:
        return 0, 0, 0
    
    bigboy = np.argmax(labels_count_cs[1:]) + 1
    bigboy_area = labels_count_cs[bigboy]

    return labels_cs, bigboy, bigboy_area


def label_nucleation(data_lc, time_coord, length_coords):
    """
    extract label of nucleation island
    """

    labels_lc, _ = ndimage.label(data_lc, np.ones((3, 3)))
    
    if length_coords[0][0] == 0:
        l1 = 0
    else:
        l1 = length_coords[0][0] - 1
    
    if length_coords[0][-1] == 200:
        l2 = 200
    else:
        l2 = length_coords[0][-1] + 1
    
    nucleation_label = max(labels_lc.T[time_coord][l1 : l2 + 1])
    length_coord = np.where(labels_lc.T[time_coord] == nucleation_label)[0][0]

    return nucleation_label, labels_lc, length_coord


def locate(data_cs, data_lc):
    """
    Execute detection, followed by loacalization and characterization of LC and
    nucleation area.
    """

    # execute detection algorithm
    labels_cs, bigboy, bigboy_area = detect(data_cs)

    # abort calculation with zero area if there is no indication of a full CS
    if bigboy_area < 1500:
        return [0]

    # locate first FE segment of full CS
    for i, disline in enumerate(labels_cs.T):
        if np.any(disline == bigboy):
            length_coords = np.where(disline == bigboy)
            time_coord = i-1
            break

    # obtain last LC of nucleation
    nucleation_label, labels_lc, length_coord = label_nucleation(data_lc, time_coord, length_coords)
    nucleation_lc = np.argwhere(labels_lc.T[time_coord] == nucleation_label).ravel()

    # obtain nucleation area
    labels_comb, _ = ndimage.label(data_cs + data_lc, np.ones((3, 3)))
    nucleation_label_comb = labels_comb.T[time_coord][length_coord]
    nucleation_region_mat = labels_comb == nucleation_label_comb
    nucleation_region_mat.T[time_coord + 1 :] = False
    nucleation_region_mat = ndimage.binary_closing(nucleation_region_mat).astype(float)

    return bigboy_area, time_coord, nucleation_lc, nucleation_region_mat
