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


cov = MGL_mu_lin.Data.data_covariance
D_data = MGL_mu_lin.Data.data_vector

invcov = np.linalg.inv(np.matrix(cov))

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

