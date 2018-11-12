#-------------------------------------------------------------------------------
""" Formatting script for converting data from *.mat or *.nirs files to python
numpy arrays.This packages the  NIRs time signals and Hb concentration signals
in a Python Dict object. This Dict is saved as a Json file to be used as
input for TensorFlow operations. """

#-------------------------------------------------------------------------------
# Import necessary modules for data conversion and packaging.

import numpy as np
import scipy.io as sio
import json
import os


#-------------------------------------------------------------------------------
#  "Switches" for 20 or 40 second intervals

make_Hb_total = True
get_HbO = False
get_HbR = False
t_20 = False
t_40 = True

if t_20:
    interval = 20 #in seconds
elif t_40:
    interval = 40 #in seconds

# Python index for  desired tone categories [1, 2, and 4].
tones = [0,1,2]

# ------------------------------------------------------------------------------
# Load .mat files
# Change '/FNA1_*' to select desired data set.
file_dir = "../fNIRS_data/" # ../ is used to 'go back' a directory
                                          # requires 'fNIRS_data' to be same
                                          # level as 'code'
mats = []
for file in os.listdir( file_dir ) :
    mats.append( sio.loadmat( file_dir+file ) )
mats = mats[:-2]
# ------------------------------------------------------------------------------
# Parse .mat files into np arrays

# These lines are the general gist of the following list comprehensions
    # s = mat_file['procResult']['s']
    # t =  mat_file['t']
    # dc = mat_file['procResult']['dc']

s = np.array([mat['procResult']['s'][0][0].astype('<f8') for mat in mats])            # [0][0] to strip .mat object packaging.
t = np.array([mat['t'][0] for mat in mats])                                           # 't' not in 'procResult', doesn't need astype.
HbO = np.array([mat['procResult']['dc'][0][0].astype('<f8')[::,0] for mat in mats])    # convert to 'float64' dtype for TensorFlow comp.
                                                                                      # [::,0] selects HbO category of hemoglobin signals
HbR = np.array([mat['procResult']['dc'][0][0].astype('<f8')[::,1] for mat in mats])

if make_Hb_total:
    dc = np.array([np.hstack((HbO[ix], HbR[ix])) for ix in range(HbO.shape[0])])
    title = "HbO_HbR_"
elif get_HbO:
    dc = HbO
    title = "HbO"
elif get_HbR:
    dc = HbR
    title = "HbR"
# Parse time categories to map labels to dc values

    # (1) Create filter of tone category onsets and offsets
    # this will be used to find start and stop rows in Hb data matrix


                            ## was [:,4:8]
filt = np.array([np.nonzero(s_i[:,4:8]) for s_i in s])                       # find rows with tone initiation; only want columns of single tone blocks (#'s 5-8)
filt_idx= np.array([np.empty([f_ix[0].shape[0], 3]).astype(int) for f_ix in filt])

for ix in range(filt_idx.shape[0]):
    filt_idx[ix][:,0] = filt[ix][0]                                               # clumn indicating tone onset
    filt_idx[ix][:,2] = filt[ix][1]                                               # column indicating tone type
    #  Following line is list comprehension to find rown with tone offset
    #  t[filt_idx[:,0]['some index']] evaluates with the time value at 'some index' in t, stored in the first column of filt_idx.
    filt_idx[ix][:,1] = [(np.abs(t[ix] - (t[ix][filt_idx[ix][:,0]][idx]+interval))).argmin()
                        for idx in range(filt_idx[ix].shape[0])]

# Create dict with tone category index as key and  list of rows indicating category onset as values
# [:,0:2:] to select the row indices in columns 0 and 1  (use 0:2, python index selection from i to N-1)
#  .tolist() method used to convert np.array to python list.  Needed for Json storage.
dc_dct = {}
for tone in tones:
    tone_filt = [filt_idx_i[filt_idx_i[:,2]==tone][:,0:2:] for filt_idx_i in filt_idx]
    dc_dct[str(tone)] = [[dc[ix][row[0]:row[1]+1].tolist() for row in rows] for ix, rows in enumerate(tone_filt)]

# -------------------------------------------------------------------------------
# Save tone_dc_dict as json for easy reading/opening by NN.
with open("../data/tone_" + title + "_dict_" + str(interval) + "_sec.json", 'w') as f:
         json.dump(dc_dct, f)
