import numpy as np
import h5py as hdf
import traceback
import re

from DNMR.fileops_loaders.data_struct import data_struct, hdf_to_dict

def read_hdf_v100(file):
    toplevel = file.keys()
    points = []
    point_indices = []
    point_numbers = []
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
    special_group_keys = [ 'detectors', 'environment']
    
    def strip_key(k):
        if(k[:5] == 'tnmr_'):
            return k[5:]
        return k
        
    def parse_dataset_into_struct(parent_group, key, struct, placement_index, key_to_write=None):
        '''Takes in an HDF group (parent_group), a _dataset_ value's key, and writes it into the appropriate place in data_struct (struct) at placement_index.'''
        kstrip = strip_key(key)
        val = parent_group[key]
        
        #handle non-array data.
        try:
            if(val.ndim == 0):
                if(isinstance(val.dtype, np.dtypes.BytesDType)):
                    conv = np.array(val, 'S').tobytes().decode('utf-8') # for strings. See HDF docs.
                    val = [conv]
                else:
                    val = [val]
        except: # in the case of dictionaries/further groups.
            pass
        
        struct[(key if key_to_write is None else key_to_write)][placement_index] = val
    
    def get_formatted_key(k):
        ks = k
        group = ''
        if('/' in ks):
            group = ks.split('/')[0]
            ks = '/'.join(ks.split('/')[1:])
        ks = strip_key(ks)
        
        for i in special_group_keys:
            if i in group:
                return i+'_'+ks
        return ks
    
    # load the first one, to get sizes etc.
    for ikey, ival in file[points[0]].items():
        if(isinstance(ival, hdf.Dataset)):
            kstrip = strip_key(ikey)
            data[kstrip] = [ [0] ] * len(points)
        elif(isinstance(ival, hdf.Group)):
            for key, val in ival.items():
                kstrip = strip_key(key)
                data[get_formatted_key(ikey+'/'+key)] = [ [0] ] * len(points)
    
    data['size'] = len(points)
    
    # Grab all data in entries, write it into data struct.
    for i, index in zip(points, point_indices):
        for ikey, ival in file[i].items():
            if(isinstance(ival, hdf.Dataset)):
                parse_dataset_into_struct(file[i], ikey, data, index)
            elif(isinstance(ival, hdf.Group)):
                for key, val in ival.items():
                    parse_dataset_into_struct(ival, key, data, index, key_to_write=get_formatted_key(ikey + '/' + key))
        
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
    
    if('actual_num_acqs' in data.params.keys()):
        data['reals'] /= data.params.actual_num_acqs[:,None]
        data['imags'] /= data.params.actual_num_acqs[:,None]
                
    #print('#' * 100)
    print(data)
    
    return data