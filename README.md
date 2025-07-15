# Cross-slipping dislocations
This repo contains the core parts of my cross-slipping dislocations research. Auxilray parts, including data across the pipline, remain hosted elsewhere.

## Research calculations workflow
1. Design atom scale simulations
2. Execute parallel computations
3. Parse and transform resulting xyz coordinates data into graph data using OVITO
4. Extract via OVITO's API the dislocations analysis data into numpy *.npy data files
5. Analyse and visualize

## Simulation
Here is an example of an atomistic [simulation](https://drive.google.com/file/d/1NZGFZTB-rZB4jdqFXhtY3P1qpQ2ZGNWx/view?usp=drive_link) of dislocation cross-slipping in Al, as processed and visualised with OVITO. Atoms shown (red) are those determined to be in the HCP crystal structure, also constitute the intrinsic stacking fault that forms between the Shockley partials (green).

## Example of a processed simulation and a few statistical analyses of it![processed simulation](graphical_abstract_snapshot.png)
Briefly, the map's x-axis represents simulation time, y-axis is the simulation's cell length - acrosswhich the full dislocation (dipole) is positioned. The gray-scale represents the dissociation width, which is the distance between two green segments (known as Shockley partials) in the simulation. Cyan regions represent time and place where the dissociated dislocation recombines. Red regions represent segments for which the dislocations are dissociated on the cross-slip plane instead of the primary one. Thus the large red triangle represents the process of cross-slipping of a single dislocation in the dipole towards its counterpart dislocation, as seen in the video. The tip of that triangle is shaded from the beginning of the full cross-slip to the last recombination, for which statistics of this recombination length and others are drawn.