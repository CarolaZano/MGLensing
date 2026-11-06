import os
os.environ["OMP_NUM_THREADS"] = "1"
print(os.path.dirname(os.path.realpath(__file__)))
#os.environ["OMP_PLACES"] = "threads"
from nautilus import Prior, Sampler
import numpy as np
import MGLensing
import time
from scipy.stats import norm
import multiprocessing
from datetime import timedelta


# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    return Usvd, Delta


MGL_mu_lin = MGLensing.MGL("ini_files/pca/config_muSigma_lin_NLmodel.yaml")
MGL_GR_nl = MGLensing.MGL("ini_files/pca/config_GR.yaml")
MGL_GR_lin = MGLensing.MGL("ini_files/pca/config_GR_lin.yaml")
MGL_nDGP_nl = MGLensing.MGL("ini_files/pca/config_nDGP.yaml")
MGL_nDGP_lin = MGLensing.MGL("ini_files/pca/config_nDGP_lin.yaml")
MGL_fR_nl = MGLensing.MGL("ini_files/pca/config_fR.yaml")
MGL_fR_lin = MGLensing.MGL("ini_files/pca/config_fR_lin.yaml")


def linear_scale_cuts(dvec_nl, dvec_lin, cov):
    """ 
    Function from Dani.
    Gets the scales (and vector indices) which are excluded if we
    are only keeping linear scales. We define linear scales such that 
    chi^2_{nl - lin) <=1.
	
    This is a version that is hopefully more reliable when data are highly correlated.
	
    dvec_nl: data vector from nonlinear theory 
    dvec_lin: data vector from linear theory
    cov: data covariance. """
	
    # Make a copy of these initial input things before they are changed,
    # so we can compare and get the indices
    dvec_nl_in = dvec_nl; dvec_lin_in = dvec_lin; cov_in = cov;
	
    # Check that data vector and covariance matrices have consistent dimensions.
    if ( (len(dvec_nl)!=len(dvec_lin)) or (len(dvec_nl)!=len(cov[:,0])) or (len(dvec_nl)!=len(cov[0,:])) ):
        raise(ValueError, "in linear_scale_cuts: inconsistent shapes of data vectors and / or covariance matrix.")
		
    while(True):
		
        chi2_temp = np.zeros(len(dvec_nl))
        for i in range(len(dvec_nl)):
            delta_dvec = np.delete(dvec_nl, i) - np.delete(dvec_lin, i)
            cov_cut = np.delete(np.delete(cov,i, axis=0), i, axis=1)
            inv_cov_cut = np.linalg.pinv(cov_cut)
            chi2_temp[i] = np.dot(delta_dvec, np.dot(inv_cov_cut, delta_dvec))
            
        #Find the index of data point that is cut to produce the smallest chi2:
        ind_min = np.argmin(chi2_temp)
        print('ind_min=', ind_min)
        
        # Cut that element
        dvec_nl = np.delete(dvec_nl, ind_min)
        dvec_lin = np.delete(dvec_lin, ind_min)
        cov = np.delete( np.delete(cov, ind_min, axis=0), ind_min, axis=1)
            
        if (chi2_temp[ind_min]<=1.0):
            break

    ex_inds = [i for i in range(len(dvec_nl_in)) if dvec_nl_in[i] not in dvec_nl]
    #print('ex_inds=', ex_inds)
	
    return ex_inds

cov = MGL_mu_lin.Data.data_covariance
D_data = MGL_mu_lin.Data.data_vector
invcov = np.linalg.inv(np.matrix(cov))

param_dic = MGL_mu_lin.params_fiducial | MGL_mu_lin.params_fixed
param_dic_all, status = MGL_mu_lin.Like.Theo.check_pars(param_dic)

Cuts_data_nl = MGL_GR_nl.Like.compute_data_vector(param_dic_all)
Cuts_data_lin = MGL_GR_lin.Like.compute_data_vector(param_dic_all)

print("Finding linear scale cuts...")
linear_cuts = linear_scale_cuts(Cuts_data_nl, Cuts_data_lin, cov)
print("Linear cuts:", linear_cuts, "length:", len(linear_cuts))
print("Original data vector length:", len(Cuts_data_nl))
print("Inverse covariance matrix shape:", invcov.shape)
# turn to zero all indices of covariance matrix corresponding to linear cuts
for i in linear_cuts:
    invcov[i,:] = 0.0
    invcov[:,i] = 0.0

def log_probability_function(pars):
        param_dic = pars | MGL_mu_lin.params_fixed
        param_dic_all, status = MGL_mu_lin.Like.Theo.check_pars(param_dic)

        if status:
                D_theory = MGL_mu_lin.Like.compute_data_vector(param_dic_all) 
        else:
             return -np.inf

        Diff = (D_data - D_theory)
             

        # Cut data vector (choleski cov. matrix = I)
        Likelihood = -0.5*(np.matmul(np.matmul(Diff.T,invcov),Diff))
        return np.float64(Likelihood)



prior = Prior()
for par_i in MGL_mu_lin.params_model:
    if MGL_mu_lin.params_priors[par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MGL_mu_lin.params_priors[par_i]['p1'] , scale=MGL_mu_lin.params_priors[par_i]['p2']))
    elif MGL_mu_lin.params_priors[par_i]['type'] == 'U':
        prior.add_parameter(par_i, dist=(MGL_mu_lin.params_priors[par_i]['p1'] , MGL_mu_lin.params_priors[par_i]['p2']))
    
# Ensure the directories exist
os.makedirs('chains', exist_ok=True)
os.makedirs('chains/hdf5', exist_ok=True)

def main():
    
    sampler = Sampler(prior, log_probability_function, 
                      filepath='chains/hdf5/'+MGL_mu_lin.hdf5_name+'.hdf5', resume=MGL_mu_lin.mcmc_resume, n_live=MGL_mu_lin.mcmc_nlive, pool=MGL_mu_lin.mcmc_pool)
    start = time.time()
    print("starting")
    sampler.run(verbose=MGL_mu_lin.mcmc_verbose, discard_exploration=True, n_eff=5000)
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish-start

    np.savetxt("chains/chain_"+MGL_mu_lin.chain_name+".txt", np.c_[points, log_w, log_l], header=MGL_mu_lin.gen_output_header(), footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))
    

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()
