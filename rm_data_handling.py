from data_handling import get_nside
import numpy as np 
import healpy as hp 
from astropy.io import fits 
import os 
from matplotlib import pyplot
def save_rm_map(output_dst, result, header, data, data_keys=[]): 
    
    nside = get_nside(data)
    output_filename = f'{output_dst}/rm_skymap_{data_keys[0]}_to_{data_keys[-1]}_ns{nside:03d}_p{header["PRNAME"]}_ot{header["OLAPTYPE"]}.fits'

    hp.write_map(output_filename,[
        result['rm_map'],
        result['rm_errs'],
        result['chi2_map'],
        result['ndata_map']
    ],overwrite=True)

    _, h = hp.read_map(output_filename, h=True)
    h = {k:v for k,v in h}
    header['NAXIS1'] = (h['NAXIS1'], '')
    header['NAXIS2'] = (h['NAXIS2'], '')
    header['BITPIX'] = (h['BITPIX'], '')
    header['LASTPIX'] = (h['LASTPIX'], '')
    header['NSIDE'] = (h['NSIDE'], '')
    for i in range(1,5):
        header[f'TFORM{i}'] = (h[f'TFORM{i}'], '')

    for data_key in data_keys:
        header.append((data_key, data[data_key]['frequency'],'GHz'))

    with fits.open(output_filename) as h: 
        h[1].header = header
        h.writeto(output_filename, overwrite=True)


def plot_rm_map(output_dst, result, data, header, data_keys=[]):
    
    nside = get_nside(data)
    output_stub = f'{output_dst}/rm_skymap_{data_keys[0]}_to_{data_keys[-1]}_ns{nside:03d}_p{header["PRNAME"]}_ot{header["OLAPTYPE"]}'

    hp.mollview(result['rm_map'],
                min=-100,
                max=100,
                title=f'RM Map: {data_keys[0]} to {data_keys[-1]}\nPrior:{header["PRNAME"]}',
                cmap=pyplot.get_cmap('RdBu_r'))
    pyplot.savefig(f'{output_stub}.pdf')
    pyplot.close()

    hp.mollview(result['rm_errs'],
                min=0,
                max=10,
                title=f'RM Errors: {data_keys[0]} to {data_keys[-1]}\nPrior:{header["PRNAME"]}',
                cmap=pyplot.get_cmap('Reds'))
    pyplot.savefig(f'{output_stub}_errs.pdf')
    pyplot.close()

    mask = result['chi2_map'] != hp.UNSEEN
    if np.sum(mask) == 0:
        vmax = 1
    else:
        vmax = np.nanpercentile(result['chi2_map'][mask],90)
    hp.mollview(result['chi2_map'],
                min=0,
                max=vmax,
                title=r'$\chi^2$'+f': {data_keys[0]} to {data_keys[-1]}\nPrior:{header["PRNAME"]}')
    pyplot.savefig(f'{output_stub}_chi2.pdf')
    pyplot.close()

    hp.mollview(result['ndata_map'],
                title=f'N Data: {data_keys[0]} to {data_keys[-1]}\nPrior:{header["PRNAME"]}')
    pyplot.savefig(f'{output_stub}_ndata.pdf')
    pyplot.close()



def create_header(data, data_keys=[], cards={}): 
    from astropy.io.fits import Header  
    from datetime import datetime 
    header = Header() 

    header['XTENSION'] = 'BINTABLE' 
    header['BITPIX'] = (None, 'array data type')
    header['NAXIS'] = (2, 'Binary table')
    header['NAXIS1'] = (None, 'Number of bytes per row') # Let healpix figure this out. 
    header['NAXIS2'] = (None, 'Number of rows') # 
    header['PCOUNT'] = (0, 'Random parameter count')
    header['GCOUNT'] = (1, 'Group count')
    header['TFIELDS'] = (4, 'Number of columns')
    header.insert('TFIELDS', ('COMMENT', ' '), after=True)
    header.insert('TFIELDS', ('COMMENT',  " *** END OF MANDATORY FIELDS *** "), after=True)
    header.insert('TFIELDS', ('COMMENT', ' '), after=True)
    header.append(('DATE',datetime.today().strftime('%Y-%m-%d'), 'Date of creation'),bottom=True) # Format 
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('COMMENT', ' *** Column Names *** '), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('TTYPE1', 'RM', 'Rotation measure'), bottom=True)
    header.append(('TTYPE2', 'RM_ERR', 'Rotation measure uncertainty'), bottom=True)
    header.append(('TTYPE3', 'CHI2', 'CHI2 of fit along line-of-sight'), bottom=True)
    header.append(('TTYPE4', 'NDATA', 'N datasets used in fit'), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('COMMENT', ' *** Column units *** '), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('TUNIT1', 'deg/m^2    ', 'Unit for RM'), bottom=True)
    header.append(('TUNIT2', 'deg/m^2    ', 'Unit for RM_ERR'), bottom=True)
    header.append(('TUNIT3', '', ''), bottom=True)
    header.append(('TUNIT4', '', ''), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('COMMENT', ' *** Column formats *** '), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('TFORM1', None, 'Type for RM'), bottom=True) # should be '1024E   ' or similar
    header.append(('TFORM2', None, 'Type for RM_ERR'), bottom=True)
    header.append(('TFORM3', None, 'Type for CHI2'), bottom=True)
    header.append(('TFORM4', None, 'Type for NDATA'), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('COMMENT', ' *** Pixel information *** '), bottom=True)
    header.append(('COMMENT', ' '), bottom=True)
    header.append(('PIXTYPE', 'HEALPIX ', 'HEALPIX pixelisation'), bottom=True)
    header.append(('ORDERING', 'RING    ', 'Pixel ordering scheme, either RING or NESTED'), bottom=True)
    header.append(('EXTNAME', 'xtension', 'Name of this binary table extension'), bottom=True)
    header.append(('FIRSTPIX', 0, 'First pixel # (0 based)'), bottom=True)
    header.append(('LASTPIX', None, 'Last pixel # (0 based)'), bottom=True)
    header.append(('NSIDE', None, 'Resolution parameter of HEALPIX'), bottom=True)
    header.append(('INDXSCHM', 'IMPLICIT', 'Indexing: IMPLICIT or EXPLICIT'), bottom=True)
    header.append(('COORDSYS', 'G', 'G = Galactic, E = ecliptic, C = celestial'), bottom=True)
    header.append(('POLCCONV', 'COSMO     ', 'Polarisation convention'), bottom=True)
    header.append(('BAD_DATA', -1.6375e+30, 'Bad data value, used to mask bad pixels'), bottom=True)
    header.append(('RELEASE', 'DR1', 'Data release version'), bottom=True)
    header.append(('ADS', 'cbass_paper_ref_here', ' '), bottom=True)
    for k,v in cards.items():
        header.append((k, v[0], v[1]),bottom=True)

    return header 

def create_output_directory(output_dst_root, data_keys, cards): 
    output_filename = f'{output_dst_root}/rm_{data_keys[0]}_to_{data_keys[-1]}'
    os.makedirs(output_filename,exist_ok=True)
    return output_filename