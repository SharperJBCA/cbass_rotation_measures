# run_rm_fits.py 
# SH - 1/9/25 
#
# Desc:
#  Overall pipeline for creating all of the different combinations of RM maps from C-BASS+S-PASS+WMAP/Planck

import numpy as np 
import healpy as hp 
import itertools 

from rm_fit_functions import fit_no_prior, fit_hutschenruter_prior, fit_ttplot_prior 
from spass_correction_functions import no_data_change, just_cbass 

from rm_data_handling import save_rm_map, plot_rm_map, create_header, create_output_directory
from data_handling import read_data, get_nside,read_hutschenruter



def main(): 

    hut_rm_map = read_hutschenruter('../../CBASS_PolarisedTF/ancillary_data/hutschenruter/faraday_sky_w_ff.hdf5')
    tt_rm_map = hp.read_map('../notebooks/qu_correlations_figures/202508/QU_low_res_rm_map.fits') 

    fit_funcs = [
        {'func':fit_no_prior,
        'prior_name':'none',
        'extra_args':{}},
        {'func':fit_hutschenruter_prior,
        'prior_name':'hutschenruter',
        'extra_args':{'data_rm_map':0.5*hut_rm_map,'RMPSIG':20.}},
        {'func':fit_ttplot_prior,
        'prior_name':'ttplot',
        'extra_args':{'data_tt_map':tt_rm_map,'RMPSIG':10.}}
    ] 
    
    data_keys_list = [
        ['cbass005','wmap023','planck030'],
        ['cbass005','wmap023','planck030','wmap030','wmap041a','wmap041b','planck044'],
        ['spass002','cbass005','wmap023','planck030','wmap030','wmap041a','wmap041b','planck044'],
        ['spass002','cbass005','wmap023','planck030'],
        ]
    
    spass_corrections = [
        {'func':no_data_change(), 
         'overlap_type': 'none'}, 
        {'func':just_cbass(),
         'overlap_type': 'just_cbass'}
    ]


    output_dst_root = 'outputs'

    for fit_info, data_keys, spass_correction in itertools.product(fit_funcs, data_keys_list, spass_corrections):

        data = read_data(
            {
                'planck':'../notebooks/planck_smoothed_maps_120arcmin.npy',
                'wmap':'../notebooks/wmap_cg_smoothed_maps_120arcmin.npy',
                'spass':'../notebooks/spass_smoothed_maps_120arcmin.npy'
            }
    ) 

        if spass_correction['func'].no_spass_data(data_keys): # no data change just returns False always 
            continue 

        data = spass_correction['func'](data) 


        
        cards = {
            'FWHM':(2.0, 'Smoothed FWHM, deg'),
            'OLAPTYPE':(spass_correction['overlap_type'],'C-BASS/S-PASS overlap'),
            'PRNAME':(fit_info['prior_name'],'Prior applied on RMs')
        }
        cards = {**cards, **{k:(v,'') for k,v in fit_info['extra_args'].items() if not 'data' in k}}
    
        output_dst = create_output_directory(output_dst_root, data_keys, cards)
        result = fit_info['func'](data, data_keys=data_keys, extra_args=fit_info['extra_args'], output_directory=output_dst) 

        header = create_header(data, data_keys=data_keys, 
                            cards=cards
        )


        plot_rm_map(output_dst, result, data, header, data_keys=data_keys)
        save_rm_map(output_dst, result, header, data, data_keys=data_keys)

if __name__ == "__main__":
    main() 