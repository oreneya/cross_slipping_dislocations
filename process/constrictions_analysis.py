import numpy as np
import sys
from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier

path_to_file, run = sys.argv[1:]

modifier = DislocationAnalysisModifier()
modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC

# save DXA data
pipeline = import_file(path_to_file)
pipeline.modifiers.append(modifier)
data = pipeline.compute()
export_file(pipeline, path_to_file[:-3] + ".ca.gz", "ca")

line_lengths = []
partials_counter = 0
infs = 0

for segment in data.dislocations.segments:
    if segment.is_infinite_line:
        infs += 1
    seg_size = np.sum(np.abs(segment.true_burgers_vector))
    if seg_size < 1:
        partials_counter += 1
    else:
        line_lengths.append(segment.length)

point_count = (partials_counter - infs) / 2 - len(line_lengths)

with open("point_count_run{}.csv".format(run), 'a') as f:
    f.writelines(str(point_count)+'\n')

with open("lines_lengths_run{}.csv".format(run), 'a') as f:
    f.writelines(str(line_lengths)+'\n')
