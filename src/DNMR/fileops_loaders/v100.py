import numpy as np
import h5py as hdf
import pytnt as tnt
import traceback
import re

from DNMR.fileops_loaders.data_struct import data_struct, hdf_to_dict

def read_hdf_v100(file):
    toplevel = file.keys()
    points = []
    point_indices = []
    point_numbers = []
    print(toplevel)
    count = 0
    for i in toplevel: # get all points
        m = re.match('entry(?P<index>[0-9]+)', i)
        if not(m is None):
            points += [ m[0] ]
            point_numbers += [ int(m['index']) ]
            point_indices += [ count ]
            count += 1
    points = np.array(points)
    point_indices = np.array(point_indices)
    point_numbers = np.array(point_numbers)
    
    sorted_indices = np.argsort(point_numbers)
    points = points[sorted_indices]
    point_numbers = point_numbers[sorted_indices]
    
    data = data_struct()
    g_keys = ['nmr_data', 'detectors', 'environment']
    # load the first one, to get sizes etc.
    for ikey, ival in file[points[0]].items():
        for key, val in ival.items():
            print(key, val)
            if(key[:5] == 'tnmr_'):
                key = key[5:]
            data[key] = [ None ] * len(points)
        
    data['size'] = len(points)

    for i, index in zip(points, point_indices):
        for ikey, ival in file[i].items():
            for key, val in ival.items():
                if(key[:5] == 'tnmr_'):
                    key = key[5:]
                data[key][index] = val
                try:
                    print(key, hdf_to_dict(val))
                except:
                    pass
        
    for key, val in data.items():
        # check if we can turn it into a dict, then numpy array
        try:
            ds = data_struct(hdf_to_dict(val[0]))
            for i in range(1, len(val)):
                ds += data_struct(hdf_to_dict(val[i]))
            data[key] = ds
        except:
            try:
                arr = np.array(val)
                try:
                    arr_f = arr.astype(float)
                    arr_i = arr.astype(int)
                    if(arr_i == arr_f):
                        arr = arr_i
                    else:
                        arr = arr_f
                except:
                    pass
                data[key] = arr
            except:
                pass
    return data