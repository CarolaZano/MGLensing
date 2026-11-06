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


MGL_q1q2q3_nl = MGLensing.MGL("ini_files/pca/config_muSigma_q1q2q3.yaml")

MGL_q1q2q3_pseudo = MGLensing.MGL("ini_files/pca/config_muSigma_q1q2q3_pseudo.yaml")

# store list of q1, q2, q3 values for the 8 theories
Model_q1q2q3_values_list = [
    (-2.0, +0.4, +1.6),
    (-2.0, +2.0, +0.8),
    (-1.2, +0.8, +1.6),
    (+0.4, +1.2, +2.0),
    (+1.2, +0.0, +0.0),
    (+1.2, +0.0, +1.6),
    (+1.2, +0.4, +1.6),
    (+1.2, +1.2, +0.8)
]


def get_B_and_M(param_dic_all, Model_q1q2q3_values_list):
    param_dic_list = []
    for q1q2q3 in Model_q1q2q3_values_list:
        param_dic_list.append(param_dic_all | {"q1": q1q2q3[0], "q2": q1q2q3[1], "q3": q1q2q3[2]})
        _, status = MGL_q1q2q3_nl.Like.Theo.check_pars(param_dic_all | {"q1": q1q2q3[0], "q2": q1q2q3[1], "q3": q1q2q3[2]})

    ### COMBINE
    # 1: find C_ell for non-linear matter power spectrum
    B_data  = np.array([MGL_q1q2q3_nl.Like.compute_data_vector(param_dic) for param_dic in param_dic_list])
    # 2: find C_ell for linear matter power spectrum
    M_data  = np.array([MGL_q1q2q3_pseudo.Like.compute_data_vector(param_dic) for param_dic in param_dic_list])
    
    return B_data, M_data


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

        # 1: find C_ell for non-linear and linear matter power spectrum
        B_data, M_data  = get_B_and_M(param_dic_all, Model_q1q2q3_values_list)

        # EXTRACT PCA MATRIX
        try:
                Usvd, Delta = findPCA(M_data, B_data, L_ch_inv)
        except:
                return -np.inf
                

        # Cut data vector (choleski cov. matrix = I)
        Diff_cut = np.matmul(Usvd[len(M_data):], Diff_ch.T)
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

    # add to header the values of q1, q2, q3 for the 8 theories
    q1q2q3_header = "################ q1, q2, q3 values for the 8 theories: ################\n"
    for i, q1q2q3 in enumerate(Model_q1q2q3_values_list):
        q1q2q3_header += "################ Theory {}: q1 = {}, q2 = {}, q3 = {}\n".format(i+1, q1q2q3[0], q1q2q3[1], q1q2q3[2])  
    np.savetxt("chains/chain_"+MGL_mu_pseudo.chain_name+".txt", np.c_[points, log_w, log_l], header=MGL_mu_pseudo.gen_output_header() + q1q2q3_header, footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()

