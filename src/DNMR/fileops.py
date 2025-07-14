
import re

import numpy as np
import h5py as hdf
import pytnt as tnt
import traceback
import re

from DNMR.fileops_loaders.data_struct import *
from DNMR.fileops_loaders.alpha import *
from DNMR.fileops_loaders.v100 import *

def get_tnt_data(fn: str):
    '''Retrieves the same data as the below function, but from a .tnt file'''
    f = tnt.TNTfile(fn)
    print('WARNING: TNT file loading is not without it\'s concerns. If you don\'t know what you\'re doing, don\'t do this!')
    input('WARNING: Mandatory input so the user is forced to read the above: ')
    
    complexes = np.swapaxes(f.DATA, 0, 1)[:,:,0,0]
    times = np.broadcast_to(f.fid_times()[None,:], complexes.shape)
    reals = np.real(complexes)
    imags = np.imag(complexes)
    
    data = data_struct()
    
    data['size'] = reals.shape[0]
    
    data['reals'] = reals
    data['imags'] = imags
    data['times'] = times*1e6
    #tnt_delay_table = [5_000_000, 2_600_000, 1_350_000, 700_000, 360_000, 190_000, 98000, 51000, 26000, 13600, 7100, 3700, 1900, 988, 512, 266, 138, 72, 37, 19, 10, 1]
    tnt_delay_table = [ 10, 19, 37, 72, 138, 266, 512, 988, 1900, 3700, 7100, 13600, 26000, 51000, 98000, 190_000, 360_000, 700_000, 1_350_000, 2_600_000, 5_000_000]
    data['sequence'] = data_struct({'0': data_struct({'relaxation_time': np.array(tnt_delay_table)})})
    data['params'] = data_struct({'post_acquisition_time': np.array([500])})
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

def get_data(fn: str):
    '''Retrieves all the data from an HDF file and stores it in a nice format.

    Parameters
    ----------
        fn: str, the filename of the data. Include file extension.

    Returns
    -------
        a dictionary, in the form { 'reals': [2d numpy array, 1st dimension is acquisition index, 2nd dimension is datapoint index],
                                    'imags': [same as reals],
                                    'times': [same as reals],  
                                    ... (other keys auto-filled!)
                                  }
    '''
    
    if(fn[-4:]=='.tnt'):
        return get_tnt_data(fn)

    with hdf.File(fn, 'r') as file:
        try:
            version_string = file.attrs['version']
        except: # no version string. Must be an alpha-version file.
            return read_hdf_valpha(file)
        
        print(f'Loading file with version string {version_string}')
        
        # else, use appropriate reader
        if(version_string in ['100']):
            return read_hdf_v100(file)
        else:
            return read_hdf_valpha(file)




