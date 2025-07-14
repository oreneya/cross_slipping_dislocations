#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outputs script
"""
size = 14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':size, 'figure.autolayout': True})
from scipy import ndimage
from nucleation import Nucleation
from constants import PIXELS
import sys
from matplotlib.patches import ConnectionPatch

# plot simulation
def plot_ax(ax, data, nuc, top_or_bottom):
    """
    Overlay LC, FE, and FL on top of dissociation widths, and show the CS
    nucleation event.
    """

    # widths
    data = np.clip(data, a_min=0, a_max=np.max(data))
    ax.imshow(data.T[0], cmap='gray', aspect='auto')
    ax.set_frame_on(False)
    ax.tick_params(axis='both')

    # overlays
    for i, color in enumerate(['cool_r', 'bwr', 'autumn']):
        ax.imshow(data.T[i+1], cmap=color, alpha=data.T[i+1], aspect='auto')

    # nucleation
    if isinstance(nuc.top_or_bottom, str):
        if nuc.top_or_bottom == top_or_bottom:

            # save / update nucleation properties to file
            save_nucleation_table(nuc)
            save_nucleation_matrix(nuc)

            # mark nucleation
            ax.plot([nuc.time, nuc.time],
                    [nuc.segment[0], nuc.segment[1]],
                    '--', color='orange', alpha=0.9, lw=0.8)
            ax.imshow(nuc.region, cmap='bwr_r', alpha=0.5*nuc.region, aspect='auto')
    

def plot_zoom(ax, data, nuc, top_or_bottom):
    """Plot zoom section of nucleation region"""

    data = np.clip(data, a_min=0, a_max=np.max(data))
    extent = (0, data.shape[0], 0, data.shape[1])

    submesh, tmin, tmax, lmin, lmax = get_submesh(data, nuc.region)
    axin = ax.inset_axes(bounds=[0.05, 0.5, 0.4, 0.4], xlim=(tmin, tmax), ylim=(lmax, lmin), xticks=[], yticks=[], xticklabels=[], yticklabels=[])

    # widths
    axin.imshow(data.T[0], extent=extent, cmap='gray', aspect='auto', origin='lower')
    
    # overlays
    for i, color in enumerate(['cool_r', 'bwr', 'autumn']):
        axin.imshow(data.T[i+1], extent=extent, cmap=color, alpha=data.T[i+1], aspect='auto', origin='lower')
    
    # nucleation
    if isinstance(nuc.top_or_bottom, str):
        if nuc.top_or_bottom == top_or_bottom:
            # mark nucleation
            lct = [nuc.time, nuc.time]
            lcl = [nuc.segment[0], nuc.segment[1]]
            axin.plot(lct, lcl, '--', color='orange', lw=0.8)
            axin.imshow(nuc.region, cmap='bwr_r', alpha=0.5*nuc.region, aspect='auto', origin='lower')

    rect = (tmin, lmin, tmax-tmin, lmax-lmin)
    #ax.indicate_inset_zoom(axin, edgecolor="black")
    ax.indicate_inset(rect, edgecolor="black", alpha=1)
    cp1 = ConnectionPatch(xyA=(tmin, lmax), xyB=(0, 0), axesA=ax, axesB=axin, coordsA="data", coordsB="axes fraction")
    cp2 = ConnectionPatch(xyA=(tmax, lmin), xyB=(1, 1), axesA=ax, axesB=axin, coordsA="data", coordsB="axes fraction")
    ax.add_patch(cp1)
    ax.add_patch(cp2)


def save_nucleation_table(nuc):
    """Save / update nucleation properties."""

    try:
        df = pd.read_csv('nucleations.csv')
    except:
        df = pd.DataFrame(columns=('run_id', 'lc_length', 'nucleation_area'), index=None)

    lc_length = nuc.segment[1] - nuc.segment[0] + 1
    nucleation_area = int(np.sum(nuc.region))

    if RUN in df.run_id.values:
        df.loc[df.run_id == RUN, 'lc_length'] = lc_length
        df.loc[df.run_id == RUN, 'nucleation_area'] = nucleation_area
    else:
        new_df = pd.DataFrame({'run_id': [RUN],
                               'lc_length': [lc_length],
                               'nucleation_area': [nucleation_area]},
                              index=None)
        df = df.append(new_df, ignore_index=True)

    df.to_csv('nucleations.csv', index=None)


def save_nucleation_matrix(nuc):
    """Save nucleation region using the LC+FE matrix.
    This is required only for auxillary analyses."""

    np.save(f"nucleation_matrix_run_{RUN}.npy", nuc.region)


def margin(position):
    """Calculate margin from periodic boundaries."""

    if position >= PIXELS / 2:
        return PIXELS - position

    return position


def extract_max_lc(time_arr):
    """Extract max LC length from either top or bottom dislocation at a given timestep."""

    try:
        island_labels = np.unique(time_arr)[1:].reshape(-1, 1)
        max_lc = max(np.sum(time_arr == island_labels, axis=1))
    except:
        max_lc = 0

    return max_lc


def plot_max_lc(ax, max_lc, attempts):
    """Plots a panel similar to the original 1-D analysis."""

    # line plot for max-length LC
    ax.plot(max_lc, 'k-', lw=1)

    # ylabel
    ax.set_ylabel("$L_{LC,max}\ (b)$")

    # CS markings    
    for (start, end, _) in attempts.T:
        ax.fill_betweenx([0, max(max_lc)], start, end, facecolor='red', alpha=0.5)

    # visuals
    ax.grid(True, which='both', color='lightgray', linestyle='--')
    ax.minorticks_on()
    ax.tick_params(axis='both')


def list_max_lc(mesh):
    """Calculates maximum LC lengths"""
    lc_label, _ = ndimage.label(mesh, np.ones((3, 3)))
    max_lc = np.empty(lc_label.shape[1])
    for i, sim_time in zip(range(len(max_lc)), lc_label.T):
        max_lc[i] = extract_max_lc(sim_time)
    return max_lc


def list_cs_attempts(mesh, max_lc):
    """Obtains the time windows when CS is present."""
    is_cs = np.sum(mesh, axis=1) > 0
    cs_islands, _ = ndimage.label(is_cs)
    cs_islands_count = np.bincount(cs_islands)
    attempts = []
    for isle_id, isle_size in enumerate(cs_islands_count[1:]):
        start = np.where(cs_islands == isle_id+1)[0][0]
        max_lc_attempt = max_lc[start-1]
        end = start + isle_size
        attempts.append([start, end, max_lc_attempt])
    return np.array(attempts).T


def save_attempts(attempts, top_or_bot, fullCS):
    """Save CS attempts data."""

    # save only if there were attempts
    if len(attempts):
        df = pd.DataFrame(columns=['start', 'end', 'max_LC_length'])
        df.start = attempts[0]
        df.end = attempts[1]
        df.max_LC_length = attempts[2]
        if fullCS:
            df.to_csv(f'attempts_run{RUN}_{top_or_bot}_fullCS')
        else:
            df.to_csv(f'attempts_run{RUN}_{top_or_bot}')


def get_nucleation_matrix_coords(nuc_mat):
    """Get the nucleation matrix time & length limits here to keep nucleation.py untouched, it's just a one-time script."""
    length_axis = nuc_mat.nonzero()[0]
    time_axis = nuc_mat.nonzero()[1]

    return time_axis.min(), time_axis.max(), length_axis.min(), length_axis.max()


def get_submesh(mesh, nuc_mat):
    """Define a sub mesh to represent the zoom section"""
    tmin, tmax, lmin, lmax = get_nucleation_matrix_coords(nuc_mat)

    # extened the submesh for better visualization
    #tmin -= int((tmax - tmin) / 2)
    _tmin_ = tmin - (tmax - tmin) * 1
    _tmax_ = tmax + (tmax - tmin) * 1
    _lmin_ = lmin - (lmax - lmin) * 1
    _lmax_ = lmax + (lmax - lmin) * 1

    return mesh[_tmin_:_tmax_, _lmin_:_lmax_], _tmin_, _tmax_, _lmin_, _lmax_


RUN = int(sys.argv[1])

# load data
top = np.load(f"mesh_top_run{RUN}.npy")
bot = np.load(f"mesh_bottom_run{RUN}.npy")

# extract nucleation event
nuc = Nucleation(top, bot)
nuc.location()

# recalculate for shifted periodic boundaries in case location detected near them.
if isinstance(nuc.top_or_bottom, str):
    margin_minimal = min([margin(nuc.segment[0]),
                        margin(nuc.segment[1])])
    if margin_minimal < PIXELS / 10:

        # shift half image over periodic boundaries
        top = ndimage.shift(top, [0, PIXELS / 2, 0], order=0, mode='grid-wrap')
        bot = ndimage.shift(bot, [0, PIXELS / 2, 0], order=0, mode='grid-wrap')

        # re-extract nucleation event (overwrite object)
        nuc = Nucleation(top, bot)
        nuc.location()

# calculate max LC lengths
max_lc_top = list_max_lc(top.T[1])
max_lc_bot = list_max_lc(bot.T[1])

# obtain attempts time windows & leading LC length
attempts_top = list_cs_attempts(top.T[2].T, max_lc_top)
attempts_bot = list_cs_attempts(bot.T[2].T, max_lc_bot)

# plot 1-D & spacetime for cross-slipping dislocation only
fig, ax = plt.subplots()

if nuc.top_or_bottom == 'top':
    data = top
    max_lc = max_lc_top
    attempts = attempts_top
else:
    data = bot
    max_lc = max_lc_bot
    attempts = attempts_bot

#plot_max_lc(axs[0], max_lc, attempts)
plot_ax(ax, data, nuc, nuc.top_or_bottom)

# zoom
#plot_zoom(ax, data, nuc, nuc.top_or_bottom)

# plot the rest
new_ticks = (ax.get_xticks() / 100).astype(int)
ax.set_xticklabels(new_ticks)
plt.xlabel('Time (ps)', fontsize=size-1)

#plt.tick_params(labelleft = False)
plt.ylabel("Position on dislocation (b)")

plt.savefig(f"constrictions_spacetime_CSonly_run{RUN}_singlepanel.png", dpi=300)
plt.savefig(f"constrictions_spacetime_CSonly_run{RUN}_singlepanel.jpeg", dpi=300)

#plt.close()
plt.show()
