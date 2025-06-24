
import re

import numpy as np
import h5py as hdf
import pytnt as tnt
import traceback
import re

class data_struct():
    data = None
    
    def __init__(self, init=None):
        self.data = {}
        
        if not(init is None):
            for k, v in init.items():
                self[k] = v
        
    def __getattr__(self, attr):
        try:
            return self.data[attr]
        except:
            return getattr(self.data, attr)
        
    def __getitem__(self, attr):
        return self.data[attr]
        
    def __setitem__(self, attr, val):
        if(isinstance(val, dict)):
            self.data[attr] = data_struct(val)
        else:
            self.data[attr] = val
        
    def __iadd__(self, r):
        for key in list(r.keys()):
            if not(key in self.data.keys()):
                self.data[key] = r[key]
                continue
            if(key == 'size'):
                self.data['size'] += r[key]
                continue
            val = self.data[key]
            if(isinstance(self.data[key], data_struct)):
                self.data[key] += r[key]
            else:
                try:
                    val = np.array(val)
                    setval = np.array(r[key])
                    if(val.ndim == 0):
                        val = np.array([val])
                    if(setval.ndim == 0):
                        setval = np.array([setval])
                    self.data[key] = np.append(val, setval, axis=0) # if numpy
                except:
                    traceback.print_exc()
                    self.data[key] += r[key] # they're lists if not.
        return self
        
    def __repr__(self):
        s = 'DATA_STRUCT\n'
        for key, val in self.data.items():
            if(isinstance(val, data_struct)):
                s += f'{key}: '+'{\n'
                s += '\t' + '\n\t'.join(val.__repr__().split('\n'))
                s += '\n}\n'
            else:
                s += f'{key}: {val.__repr__()}\n'
        s += 'END DATA_STRUCT'
        return s

def get_tnt_data(fn: str):
    '''Retrieves the same data as the below function, but from a .tnt file'''
    f = tnt.TNTfile(fn)
    
    complexes = np.swapaxes(f.DATA, 0, 1)[:,:,0,0]
    times = np.broadcast_to(f.fid_times()[None,:], complexes.shape)
    reals = np.real(complexes)
    imags = np.imag(complexes)
    
    data = data_struct()
    
    data['size'] = reals.shape[0]
    
    data['reals'] = reals
    data['imags'] = imags
    data['times'] = times*1e6
    tnt_delay_table = [5_000_000, 2_600_000, 1_350_000, 700_000, 360_000, 190_000, 98000, 51000, 26000, 13600, 7100, 3700, 1900, 988, 512, 266, 138, 72, 37, 19, 10, 1]
    data['sequence'] = data_struct({'0': data_struct({'relaxation_time': tnt_delay_table})})
    #data['relaxation_times'] = np.array([5_000_000, 2_600_000, 1_350_000, 700_000, 360_000, 190_000, 98000, 51000, 26000, 13600, 7100, 3700, 1900, 988, 512, 266, 138, 72, 37, 19, 10, 1])
    #tmp = np.copy(data['relaxation_times'])
    #for i in range(len(tmp)):
    #    data['relaxation_times'][-i-1] = tmp[i]
    
    # TODO
    magf = re.search('.*?(?P<magfield>\\d+([.]\\d+)?)Oe.*?', fn)
    freq = re.search('.*?(?P<freq>\\d+([.]\\d+)?)MHz.*?', fn)
    if(magf):
        data['ppms_mf'] = np.broadcast_to(np.array([float(magf['magfield'])]), (data['size'],))
    if(freq):
        data['obs_freq'] = np.broadcast_to(np.array([float(freq['freq'])]), (data['size'],))
    
    return data

def hdf_to_dict(g): # takes group, gives dict
    def t(n, g, d):
        ds = d
        for i in n.split('/')[:-1]:
            ds = ds[i]
        if(isinstance(g, hdf.Group)):
            ds[n.split('/')[-1]] = data_struct(hdf_to_dict(g))
        else:
            try:
                tmp = np.array(g)
            except:
                tmp = g
            ds[n.split('/')[-1]] = tmp
    d = {}
    g.visititems(lambda a,b: t(a,b,d))
    
    return d

def get_data(fn: str):
    '''Retrieves all the data from an HDF file and stores it in a nice format.

    Parameters
    ----------
        fn: str, the filename of the data. Include file extension.

    Returns
    -------
        a dictionary, in the form { 'reals': [2d numpy array, 1st dimension is sequence index, 2nd dimension is datapoint index],
                                    'imags': [same as reals],
                                    'times': [same as reals],  
                                    ... (other keys auto-filled!)
                                  }
    '''
    
    if(fn[-4:]=='.tnt'):
        return get_tnt_data(fn)

    with hdf.File(fn, 'r') as file:
        toplevel = file.keys()
        points = []
        point_indices = []
        for i in toplevel: # get all points
            m = re.match('point(?P<index>[0-9]+)', i)
            if not(m is None):
                points += [ m[0] ]
                point_indices += [ int(m['index']) ]
        points = np.array(points)
        point_indices = np.array(point_indices)
        
        sorted_indices = np.argsort(point_indices)
        points = points[sorted_indices]
        point_indices = point_indices[sorted_indices]
        
        data = data_struct()
        # load the first one, to get sizes etc.
        for key, val in file[points[0]].items():
            if(key[:5] == 'tnmr_'):
                key = key[5:]
            data[key] = [ None ] * len(points)
            
        data['size'] = len(points)

        for i, index in zip(points, point_indices):
            for key, val in file[i].items():
                if(key[:5] == 'tnmr_'):
                    key = key[5:]
                # legacy logic
                if(key == 'relaxation_time'):
                    key = 'delay_time'
                # end of legacy logic
                data[key][index] = val
        
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
                    data[key] = arr
                except:
                    pass
        print(data)
        data['times'] = data['times'][:,:1024] # also legacy.
    return data





