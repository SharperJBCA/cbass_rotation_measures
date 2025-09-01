from fit_rm_mcmc import fit_rm_mcmc 
import numpy as np 
import healpy as hp 
import copy 

def create_angle_maps(data):
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

    return data 

class top_hat_prior:

    def __init__(self, index, min=-np.inf, max=np.inf):
        self.index = index 
        self.min = min 
        self.max = max 

    def __call__(self, x):
        x_0 = x[self.index] 
        if self.min < x_0 < self.max:
            return 0 
        else:
            return -np.inf 
        
class gaussian_prior:

    def __init__(self, index, mean=0, sigma=np.inf):
        self.index = index 
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x):
        x_0 = x[self.index] 
        return -0.5 * (x_0 - self.mean)**2/self.sigma**2
    

class amplitude_prior:

    def __init__(self, index, mean_map=0, sigma=np.inf):
        self.index = index 
        self.mean_map = mean_map
        self.sigma = sigma

    def __call__(self, x):
        x_0 = x[self.index] 
        return -0.5 * (x_0 - self.mean)**2/self.sigma**2


class lnlike:

    def __init__(self, priors=[]):

        self.priors=[]

    def lnlike(self, x, templates, data, errs):
        model = np.dot(templates,x)
        lnln = -0.5*np.sum((data - model)**2/errs**2)

        lnprior = np.sum([p(x) for p in self.priors])
        return lnln + lnprior 

    def __call__(self, x, templates, data, errs):
        return self.lnlike(x, templates, data, errs)


def fit_no_prior(data_dict, data_keys=[], extra_args=[]): 

    data_subset = {k:data_dict[k] for k in data_keys} 

    data_subset = create_angle_maps(data_subset)

    angle_maps = np.vstack([data_subset[map_name]['angles'] for map_name in data_keys])
    angle_err_maps = np.vstack([data_subset[map_name]['angle_stddev'] for map_name in data_keys])
    frequencies = np.array([data_subset[map_name]['frequency'] for map_name in data_keys])
    nwalkers = 100
    nsteps =  5000
    nburnin = 1000
    nthin = 15

    npix = len(data_dict[data_keys[0]]['I_smth'])
    lnlikes = [lnlike() for ipix in range(npix)] 

    rm_map, rm_errs, chi2_map, output_angle_map, output_angle_err_map, ndata_map, angle_offset_map = fit_rm_mcmc(
            angle_maps,
            angle_err_maps,
            frequencies,
            lnlikes, # likelihood function for each pixel 
            nburnin=nburnin,
            nthin=nthin,
            nsteps=nsteps,
            nwalkers=nwalkers)

    return {
        'rm_map':rm_map,
        'rm_errs':rm_errs,
        'chi2_map':chi2_map,
        'ndata_map':ndata_map
    }

def fit_hutschenruter_prior(data_dict, data_keys=[], extra_args=[]): 

    hutschenruter_map, = extra_args

    data_subset = {k:data_dict[k] for k in data_keys} 

    data_subset = create_angle_maps(data_subset)

    angle_maps = np.vstack([data_subset[map_name]['angles'] for map_name in data_keys])
    angle_err_maps = np.vstack([data_subset[map_name]['angle_stddev'] for map_name in data_keys])
    frequencies = np.array([data_subset[map_name]['frequency'] for map_name in data_keys])
    nwalkers = 100
    nsteps =  5000
    nburnin = 1000
    nthin = 15

    npix = len(data_dict[data_keys[0]]['I_smth'])
    lnlikes = [lnlike(
        priors=[gaussian_prior(1,mean=-hutschenruter_map[ipix],sigma=10.0)]
        ) for ipix in range(npix)] 

    rm_map, rm_errs, chi2_map, output_angle_map, output_angle_err_map, ndata_map, angle_offset_map = fit_rm_mcmc(
            angle_maps,
            angle_err_maps,
            frequencies,
            lnlikes, # likelihood function for each pixel 
            nburnin=nburnin,
            nthin=nthin,
            nsteps=nsteps,
            nwalkers=nwalkers)

    return {
        'rm_map':rm_map,
        'rm_errs':rm_errs,
        'chi2_map':chi2_map,
        'ndata_map':ndata_map
    }

def fit_ttplot_prior(data_dict, data_keys=[], extra_args=[]): 

    ttplot_map, = extra_args

    data_subset = {k:data_dict[k] for k in data_keys} 

    data_subset = create_angle_maps(data_subset)

    angle_maps = np.vstack([data_subset[map_name]['angles'] for map_name in data_keys])
    angle_err_maps = np.vstack([data_subset[map_name]['angle_stddev'] for map_name in data_keys])
    frequencies = np.array([data_subset[map_name]['frequency'] for map_name in data_keys])
    nwalkers = 100
    nsteps =  5000
    nburnin = 1000
    nthin = 15

    npix = len(data_dict[data_keys[0]]['I_smth'])
    lnlikes = [lnlike(
        priors=[gaussian_prior(1,mean=-ttplot_map[ipix],sigma=10.0)]
        ) for ipix in range(npix)] 

    rm_map, rm_errs, chi2_map, output_angle_map, output_angle_err_map, ndata_map, angle_offset_map = fit_rm_mcmc(
            angle_maps,
            angle_err_maps,
            frequencies,
            lnlikes, # likelihood function for each pixel 
            nburnin=nburnin,
            nthin=nthin,
            nsteps=nsteps,
            nwalkers=nwalkers)

    return {
        'rm_map':rm_map,
        'rm_errs':rm_errs,
        'chi2_map':chi2_map,
        'ndata_map':ndata_map
    }