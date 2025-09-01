import toml
import logging
import os
import numpy as np
import healpy as hp
import matplotlib 
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from tqdm import tqdm

import emcee 
import corner 

import multiprocessing as mp
from functools import partial

def plot_errorbar(x,y,yerr,model,output_figure_file, ipix, nside):
    theta,phi = hp.pix2ang(nside, ipix) 
    gb, gl = (np.pi/2. - theta)*180./np.pi, phi*180./np.pi
    os.makedirs(os.path.dirname(output_figure_file),exist_ok=True)

    plt.errorbar(x,y,yerr=yerr,fmt='o')
    plt.plot(x,model,'r--')
    plt.xlabel(r'$\lambda^2$ [m$^2$]')
    plt.ylabel(r'$\phi$ [degrees]')
    plt.title(f'RM Fit {ipix}')
    plt.text(0.1,0.9,f'GL: {gl:.2f} GB: {gb:.2f} ',transform=plt.gca().transAxes)
    plt.savefig(f'{output_figure_file}',dpi=300)
    plt.close()

def plot_corner(flat_samples,labels,output_figure_file):
    os.makedirs(os.path.dirname(output_figure_file),exist_ok=True)
    fig = corner.corner(
        flat_samples, labels=labels)
    fig.savefig(output_figure_file)
    plt.close()



    
def log_prior(x, hmap, herr):
    """Prior on RM from hutschenruter map"""
    if (x[1] < -20000) | (x[1] > 20000):
        return -np.inf
    return -0.5*(x[1] - hmap)**2/herr**2

def lnlike(x, templates, data, errs,prior=True):
    model = np.dot(templates,x)

    lnprior = 0 # log_prior(x,hmap,herr)
    lnln = -0.5*np.sum((data - model)**2/errs**2)
    if prior:
        return lnln + lnprior
    else:
        return lnln

def get_pixel_info(pol_angle_maps, pol_angle_errs, frequencies, wavelengths_orig, rotate=True): 

    npix = len(pol_angle_maps[0])
    pixels = np.arange(npix,dtype=int)

    all_angles = []
    all_errors = [] 
    all_wavelengths = []
    all_freqs = []

    for i,ipix in enumerate(tqdm(pixels)):

        # Get the pixel data
        angles = np.array([angle[ipix] for angle in pol_angle_maps])
        errs   = np.array([err[ipix] for err in pol_angle_errs])
        wavelengths = wavelengths_orig*1
        mask = (angles > -1e20) & np.isfinite(angles) & (errs > 0) & np.isfinite(errs)
        angles = angles[mask]
        errs = errs[mask]
        freqs = frequencies[mask]
        wavelengths = wavelengths[mask]

        # Fix edge wrapping effects  
        if rotate:
            freq_idx = np.argsort(freqs)
            for idx_1,idx_2 in zip(freq_idx[:-1],freq_idx[1:]):
                diff = angles[idx_2] - angles[idx_1]
                if diff > np.pi/2.:
                    angles[idx_2] -= np.pi
                elif diff < -np.pi/2.:
                    angles[idx_2] += np.pi

            if angles[freq_idx[-1]] > np.pi/2.:
                angles -= np.pi
            elif angles[freq_idx[-1]] < -np.pi/2.:
                angles += np.pi

        all_angles.append(angles)
        all_errors.append(errs)
        all_wavelengths.append(wavelengths)
        all_freqs.append(freqs)

    return all_angles, all_errors, all_wavelengths, all_freqs

def process_pixel(pixel_data, nburnin, nthin, nwalkers, nsteps, return_chains=False):
    """
    Process a single pixel for MCMC fitting.
    """
    ipix, angles, errs, wavelengths, freqs, lnlike = pixel_data

    #if ipix != 20:
    #    return  ipix, ([hp.UNSEEN,hp.UNSEEN], [hp.UNSEEN,hp.UNSEEN], hp.UNSEEN, hp.UNSEEN, hp.UNSEEN, hp.UNSEEN)
    #if len(angles) <= 3:
    #    return ipix, ([hp.UNSEEN,hp.UNSEEN], [hp.UNSEEN,hp.UNSEEN], hp.UNSEEN, hp.UNSEEN, hp.UNSEEN, hp.UNSEEN)
    #if ipix != 300:
    #    return ipix, ([hp.UNSEEN,hp.UNSEEN],[hp.UNSEEN,hp.UNSEEN], hp.UNSEEN, hp.UNSEEN, hp.UNSEEN, hp.UNSEEN)
    # Skip conditions
    if len(angles) < 2 or np.min(freqs) > 10:
        return ipix, ([hp.UNSEEN,hp.UNSEEN],[hp.UNSEEN,hp.UNSEEN], hp.UNSEEN, hp.UNSEEN, hp.UNSEEN, hp.UNSEEN)
    
    templates = np.ones((len(wavelengths), 2))
    templates[:,1] = wavelengths**2 
    ndim = templates.shape[1]
    
    pmdl = np.poly1d(np.polyfit(wavelengths**2, angles, 1, w=1./errs))
    # Initialize positions
    pos = 1e-4 * np.random.randn(nwalkers, ndim)
    pos[:,0] += angles[np.argmax(np.abs(freqs))]
    pos[:,1] += pmdl[1]
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike, args=(templates, angles, errs))
    sampler.run_mcmc(pos, nsteps, progress=False)  # Disable progress bar for parallel processing
    flat_samples = sampler.get_chain(discard=nburnin, thin=nthin, flat=True)
    best_fit = np.mean(flat_samples, axis=0)
    error_fit = np.std(flat_samples, axis=0)
    angle_offset = 0

    # Build chi2 from posterior distribution. 
    residuals = flat_samples[:,0,None] + flat_samples[:,1,None]*wavelengths[None,:]**2 - angles[None,:]
    residuals = residuals**2/errs[None,:]**2 
    chi2 = np.sum(residuals,axis=1) 
    chi2 = np.median(chi2)
    
    # from matplotlib import pyplot 
    # import sys 
    # ax = pyplot.subplot()
    # pyplot.errorbar(wavelengths**2, angles, yerr=errs, fmt='.')
    # pyplot.plot(wavelengths**2,best_fit[0] + best_fit[1]*wavelengths**2,'k')
    # pyplot.text(0.05,0.9, f'{best_fit[1]:.2f}', transform=ax.transAxes)
    # pyplot.savefig('test.png')
    
    if return_chains:
        return ipix, (best_fit, error_fit, chi2, len(angles), angle_offset, templates, flat_samples )

    return ipix, (best_fit, error_fit, chi2, len(angles), angle_offset, templates )

def fit_rm_mcmc(pol_angle_maps,
                pol_angle_errs,
                frequencies,
                lnlikes,
                nburnin=100,
                nthin=15,
                nwalkers=32,
                nsteps=1000,
                n_processes=None):
    """
    Parallel version of RM fitting using MCMC methods plus the Hutschenreuter map as a prior.
    
    Additional Parameters
    ----------
    n_processes : int, optional
        Number of processes to use. If None, uses cpu_count()
    """
    wavelengths_orig = speed_of_light/(frequencies*1e9)
    npix = len(pol_angle_maps[0])
    
    # Initialize output maps
    rm_map = np.full(npix, hp.UNSEEN)
    rm_errs = np.full(npix, hp.UNSEEN)
    chi2_map = np.full(npix, hp.UNSEEN)
    output_angle_map = np.full(npix, hp.UNSEEN)
    output_angle_err_map = np.full(npix, hp.UNSEEN)
    angle_offset_map = np.full(npix, hp.UNSEEN)
    ndata_map = np.full(npix, hp.UNSEEN)
    
    # Prepare pixel data
    all_angles, all_errs, all_wavelengths, all_freqs = get_pixel_info(
        pol_angle_maps, pol_angle_errs, frequencies, wavelengths_orig)
    
    # Create list of pixel data
    pixel_data = list(zip(range(npix), all_angles, all_errs, all_wavelengths, all_freqs, lnlikes))
    
    # Set up the pool
    if n_processes is None:
        n_processes =  mp.cpu_count()//2
    
    # Create partial function with fixed parameters
    process_pixel_partial = partial(
        process_pixel,
        nburnin=nburnin,
        nthin=nthin,
        nwalkers=nwalkers,
        nsteps=nsteps
    )

    
    # Process pixels in parallel
    with mp.Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_pixel_partial, pixel_data),
            total=npix,
            desc="Processing pixels"
        ))
    
    # Collect results
    #best_fit, error_fit, chi2, len(angles), angle_offset, templates
    for ipix, (best_fit, error_fit, chi2, ndata, angle_offset, templates) in results:
        if (ndata ==0) | (ndata == hp.UNSEEN):
            continue
        angle = best_fit[0]
        angle_err = error_fit[0] 
        rm = -best_fit[1] 
        rm_err = error_fit[1]
        output_angle_map[ipix] = angle
        output_angle_err_map[ipix] = angle_err
        rm_map[ipix] = rm
        rm_errs[ipix] = rm_err
        chi2_map[ipix] = chi2
        ndata_map[ipix] = ndata
        angle_offset_map[ipix] = angle_offset
    
    return rm_map, rm_errs, chi2_map, output_angle_map, output_angle_err_map, ndata_map, angle_offset_map



import read_maps 
def main(map_names, output_filename):


    data = read_maps.read_map_planck('../notebooks/planck_smoothed_maps_120arcmin.npy')
    data_wmap = read_maps.read_map_wmap_cgdr1('../notebooks/wmap_cg_smoothed_maps_120arcmin.npy')
    data_spass = np.load('../notebooks/spass_smoothed_maps_120arcmin.npy',allow_pickle=True).flatten()[0]

    for k, v in data_wmap.items():
        data[k] = v 
    for k, v in data_spass.items():
        data[k] = v 
    for k, v in data.items(): 

        angles = 0.5*np.arctan2(v['U_smth'],v['Q_smth'])
        v['angles'] = angles 
        P = np.sqrt(v['U_smth']**2 + v['Q_smth']**2)
        A = v['U_smth']**2*v['UU_smth'] + v['Q_smth']**2*v['QQ_smth']
        angle_error = np.sqrt(A)/2/P**2 
        v['angle_stddev'] = angle_error 
        mask = v['U_smth'] == hp.UNSEEN 
        v['angles'][mask] = hp.UNSEEN
        v['angle_stddev'][mask]=hp.UNSEEN


    #map_names = ['spass002','cbass005','wmap023','planck030']#,'wmap030','planck044','wmap041a','wmap041b'] 

    angle_maps = np.vstack([data[map_name]['angles'] for map_name in map_names])
    angle_err_maps = np.vstack([data[map_name]['angle_stddev'] for map_name in map_names])
    frequencies = np.array([data[map_name]['frequency'] for map_name in map_names])
    nwalkers = 100
    nsteps =  5000
    nburnin = 1000
    nthin = 15

    rm_map, rm_errs, chi2_map, output_angle_map, output_angle_err_map, ndata_map, angle_offset_map = fit_rm_mcmc(
            angle_maps,
            angle_err_maps,
            frequencies,
            nburnin=nburnin,
            nthin=nthin,
            nsteps=nsteps,
            nwalkers=nwalkers)
    
    hp.write_map(f'rm_maps/082025/{output_filename}.fits', [rm_map,rm_errs, chi2_map], overwrite=True)
    hp.write_map(f'rm_maps/082025/{output_filename}_angles.fits', [output_angle_map,output_angle_err_map, angle_offset_map], overwrite=True)
    hp.mollview(rm_map, min=-100,max=100, cmap=plt.get_cmap('RdBu_r'))
    plt.savefig(f'rm_maps/082025/{output_filename}.pdf')
    #             create_pixel_plots=create_pixel_plots,
    #               prefix = prefix,
    #             nsteps=nsteps)
    # hp.write_map(f'{output_directory}/{prefix}{herr:.0f}_mcmc_rm_map.fits',[rm_map,rm_errs,chi2_map, ndata_map],overwrite=True)
    # hp.write_map(f'{output_directory}/{prefix}{herr:.0f}_mcmc_rm_map_angles.fits',[output_angle_map,output_angle_err_map, angle_offset_map],overwrite=True)

if __name__ == "__main__":
    #main(['spass002','cbass005','wmap023','planck030'],'rm_map_2to30GHz_no_prior') 
    #main(['spass002','cbass005','wmap023','planck030','wmap030','planck044','wmap041a','wmap041b'] ,'rm_map_2to41GHz_no_prior') 
    main(['cbass005','wmap023','planck030'],'rm_map_5to30GHz_no_prior') 
    #main(['cbass005','wmap023','planck030','wmap030','planck044','wmap041a','wmap041b'] ,'rm_map_5to41GHz_no_prior') 