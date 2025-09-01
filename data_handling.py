import numpy as np 
import healpy as hp 
import h5py 
c = 299792458.
kb = 1.3806488e-23
h = 6.62606957e-34
T_cmb = 2.725
Jy = 1e26
def planckcorr(nu_in):

    nu = nu_in * 1e9
    x = h*nu/kb/T_cmb

    return x**2*np.exp(x)/(np.exp(x) - 1.)**2

def read_map_planck(numpy_save_filename):

    data = np.load(numpy_save_filename, allow_pickle=True).flatten()[0] 

    units = {'planck353':'KCMB',
            'planck217':'KCMB',
            'planck143':'KCMB',
            'planck100':'KCMB',
            'planck070':'KCMB',
            'planck044':'KCMB',
            'planck030':'KCMB',
            'cbass005':'K'}

    for k, unit in units.items(): 
        d = data[k]
        if unit == 'KCMB':
            for c in ['I_smth','Q_smth','U_smth']:
                d[c] *= planckcorr(d['frequency'])
            for c in ['II_smth','QQ_smth','UU_smth']:
                d[c] *= planckcorr(d['frequency'])**2
        if k == 'cbass005':
            gd = (d['U_smth'] != hp.UNSEEN)
            d['U_smth'][gd] *= -1
    
    return data

def read_map_wmap_cgdr1(numpy_save_filename):
  
    data_wmap = np.load(numpy_save_filename, allow_pickle=True).flatten()[0] 
    for k, v in data_wmap.items(): 
        for k2,v2 in v.items():
            if 'smth' in k2:
                v2*= planckcorr(v['frequency'])

    return data_wmap 

def read_map_spass(numpy_save_filename): 
    return np.load(numpy_save_filename,allow_pickle=True).flatten()[0]

def get_nside(data_dict):
    keys = list(data_dict.keys())
    npix = len(data_dict[keys[0]]['I_smth'])
    return hp.npix2nside(npix)  

def read_data(file_path_dict): 

    functions_dict = {
        'planck':read_map_planck,
        'wmap':read_map_wmap_cgdr1,
        'spass':read_map_spass
        }
    
    data = {} 
    for k, v in file_path_dict.items(): 
        data = {**data, **functions_dict[k](v)} 

    return data 

def read_hutschenruter(filename,nside_out=16):
    
    with h5py.File(filename) as h:
        rm_map = h['faraday sky']['mean'][...]


    return hp.ud_grade(rm_map,nside_out)