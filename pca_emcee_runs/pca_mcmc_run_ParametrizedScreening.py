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



MGL_q1q2q3_nl = MGLensing.MGL("ini_files/pca/config_muSigma_q1q2q3.yaml")
MGL_q1q2q3_pseudo = MGLensing.MGL("ini_files/pca/config_muSigma_q1q2q3_pseudo.yaml")

# find large range of q1q2q3 values
q1_arr = np.linspace(-2,2, 6)
q2_arr = np.linspace(-2,2, 6)
q3_arr = np.linspace(-2,2, 6)

# find all combinations of q1q2q3 values in the ranges above, and store in list
Model_q1q2q3_values_list = [
    (q1, q2, q3)
    for q1 in q1_arr
    for q2 in q2_arr
    for q3 in q3_arr
]

# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA_and_explainedvariance(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    # Calculate explained variance, variance_i = s_i/sum(s)
    explained_variance = s / np.sum(s)
    return Usvd, Delta, explained_variance

def get_Usvd_and_Delta(param_dic_all, Model_q1q2q3_values_list):
    
    param_dic_list = []
    for q1q2q3 in Model_q1q2q3_values_list:
        param_dic_list.append(param_dic_all | {"q1": q1q2q3[0], "q2": q1q2q3[1], "q3": q1q2q3[2]})
        #_, status = MGL_q1q2q3_nl.Like.Theo.check_pars(param_dic_all | {"q1": q1q2q3[0], "q2": q1q2q3[1], "q3": q1q2q3[2]})

    ### COMBINE
    # 1: find C_ell for non-linear matter power spectrum
    B_data  = np.array([MGL_q1q2q3_nl.Like.compute_data_vector(param_dic) for param_dic in param_dic_list])
    # 2: find C_ell for pseudo matter power spectrum
    M_data  = np.array([MGL_q1q2q3_pseudo.Like.compute_data_vector(param_dic) for param_dic in param_dic_list])

    Usvd, Delta, explained_variance = findPCA_and_explainedvariance(M_data, B_data, L_ch_inv)

    # keep PCs that explain more than 99.9% variance
    pc_indices = np.where(explained_variance > 0.001)

    return Usvd, Delta, explained_variance, pc_indices


MGL_mu_pseudo = MGLensing.MGL("ini_files/pca/config_muSigma_pseudo_PCA.yaml")

cov = MGL_mu_pseudo.Data.data_covariance
D_data = MGL_mu_pseudo.Data.data_vector

L_choleski_uncut = np.linalg.cholesky(np.matrix(cov))
L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)

L_ch_inv = L_choleski_inv_uncut



def log_probability_function(pars):
        param_dic = pars | MGL_mu_pseudo.params_fixed
        param_dic_all, status = MGL_mu_pseudo.Like.Theo.check_pars(param_dic)

        if status:
                D_theory = MGL_mu_pseudo.Like.compute_data_vector(param_dic_all) 
        else:
             return -np.inf

        Diff = (D_data - D_theory)

        # Find Choleski scaled data vector
        Diff_ch = np.array(np.matmul(L_ch_inv, Diff.T))[0]

        # EXTRACT PCA MATRIX
        try:
             Usvd, Delta, explained_variance, pc_indices = get_Usvd_and_Delta(param_dic_all, Model_q1q2q3_values_list)
        except:
             return -np.inf
             

        # Cut data vector (choleski cov. matrix = I)
        Diff_cut = np.matmul(Usvd[len(pc_indices[0]):], Diff_ch.T)
        Likelihood = -0.5*(np.matmul(Diff_cut.T,Diff_cut))
        return Likelihood



prior = Prior()
for par_i in MGL_mu_pseudo.params_model:
    if MGL_mu_pseudo.params_priors[par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MGL_mu_pseudo.params_priors[par_i]['p1'] , scale=MGL_mu_pseudo.params_priors[par_i]['p2']))
    elif MGL_mu_pseudo.params_priors[par_i]['type'] == 'U':
        prior.add_parameter(par_i, dist=(MGL_mu_pseudo.params_priors[par_i]['p1'] , MGL_mu_pseudo.params_priors[par_i]['p2']))
 
# Ensure the directories exist
os.makedirs('chains', exist_ok=True)
os.makedirs('chains/hdf5', exist_ok=True)

def main():
    
    sampler = Sampler(prior, log_probability_function, 
                      filepath='chains/hdf5/'+MGL_mu_pseudo.hdf5_name+'.hdf5', resume=MGL_mu_pseudo.mcmc_resume, n_live=MGL_mu_pseudo.mcmc_nlive, pool=MGL_mu_pseudo.mcmc_pool)
    start = time.time()
    print("starting")
    sampler.run(verbose=MGL_mu_pseudo.mcmc_verbose, discard_exploration=True, n_eff=5000)
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish-start

    np.savetxt("chains/chain_"+MGL_mu_pseudo.chain_name+".txt", np.c_[points, log_w, log_l], header=MGL_mu_pseudo.gen_output_header(), footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))
    

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()

