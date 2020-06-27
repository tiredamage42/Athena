'''
script to combine all the weight dataset loose files into a root copy that contains the whole dataset
if the combined root file exists, the loose files are appended
'''
import os
import glob
import numpy as np

final_path = 'datasets/weights_dataset.npz'

# get all the loose files
files = glob.glob('datasets/weights/*.npz')

# make sure there are loose files
assert len(files) != 0

# concatenate them all to a single numpy array
data = np.concatenate([ np.load(np_name)['arr_0'] for np_name in files ])

# if the main file exists, append the already existing data
if os.path.exists(final_path):
    data = np.concatenate([data, np.load(final_path)['arr_0']])

print ('New Dataset Length: {0}'.format(data.shape[0])) 
print ('Saving to disk... (This may take a while)')

# save the updated dataset
np.savez_compressed(final_path, data)

print ('Done! You may now delete the following weights files:')
[ print('\t{0}'.format(f) ) for f in files ]
