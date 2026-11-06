import os
os.environ["OMP_NUM_THREADS"] = "1"
print(os.path.dirname(os.path.realpath(__file__)))
#os.environ["OMP_PLACES"] = "threads"
import numpy as np
import MGLensing
import time
from scipy.stats import norm
import multiprocessing
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    return Usvd, Delta


MGL_mu_lin = MGLensing.MGL("ini_files/pca/config_muSigma_pseudo_PCA.yaml")

ells_wl_bins = []
ells_gc_bins = []
ells_xc_bins = []
#print(MGL_mu_lin.Survey.masked_data_vector_3x2pt_ells)
for i in range(MGL_mu_lin.Survey.mask_ells_wl.shape[0]):
    #print('bin ', i, ' number of ells_wl: ', sum(MGL_mu_lin.Survey.mask_ells_wl[i]))
    ells_wl_bins.append(sum(MGL_mu_lin.Survey.mask_ells_wl[i]))

ells_wl_bins = np.array(ells_wl_bins)
ells_gc_bins = np.array(ells_gc_bins)
ells_xc_bins = np.array(ells_xc_bins)

#MGL_GR_nl = MGLensing.MGL("ini_files/pca/config_GR.yaml")
#MGL_GR_lin = MGLensing.MGL("ini_files/pca/config_GR_pseudo.yaml")
MGL_nDGP_nl = MGLensing.MGL("ini_files/pca/config_nDGP.yaml")
MGL_nDGP_lin = MGLensing.MGL("ini_files/pca/config_nDGP_pseudo.yaml")
MGL_fR_nl = MGLensing.MGL("ini_files/pca/config_fR.yaml")
MGL_fR_lin = MGLensing.MGL("ini_files/pca/config_fR_pseudo.yaml")
B_models = [
    MGL_nDGP_nl,
    MGL_fR_nl
]
M_models = [
    MGL_nDGP_lin,
    MGL_fR_lin
]

# [r"$\Omega_m$", r"$w_b$",r"$h$",r"$log10As$",r"$n_s$", r"$\mu_0$",r"$a1_{IA}$",r"$\eta 1_{IA}$"]
# [ 0.31892037  0.0226738   0.67567714  3.02769177  0.95492059 -4.98800875, 1.7180302  -0.4060574  -8.00399565]
pars = {
    "Omega_m": 0.31892037,
    "Ombh2": 0.0226738,
    "h": 0.67567714,
    "log10As": 3.02769177,
    "ns": 0.95492059,
    "log10f_R0": -4.98800875,
    "a1_IA": 1.7180302,
    "eta1_IA": -0.4060574}

print("Parameters: ", pars)
cov = MGL_mu_lin.Data.data_covariance
D_data = MGL_mu_lin.Data.data_vector

L_choleski_uncut = np.linalg.cholesky(np.matrix(cov))
L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)

L_ch_inv = L_choleski_inv_uncut

param_dic = pars | MGL_mu_lin.params_fixed
param_dic_all, status = MGL_mu_lin.Like.Theo.check_pars(param_dic)

if status:
    D_theory = MGL_mu_lin.Like.compute_data_vector(param_dic_all) 
else:
    D_theory = -np.inf

Diff = (D_data - D_theory)

# Find Choleski scaled data vector
Diff_ch = np.array(np.matmul(L_ch_inv, Diff.T))[0]

### COMBINE
# 1: find C_ell for non-linear matter power spectrum
B_data  = np.array([m.Like.compute_data_vector(param_dic_all) for m in B_models])
# 2: find C_ell for linear matter power spectrum
M_data  = np.array([m.Like.compute_data_vector(param_dic_all) for m in M_models])

# EXTRACT PCA MATRIX
try:
    Usvd, Delta = findPCA(M_data, B_data, L_ch_inv)
except:
    print("Error occurred while finding PCA components.")
            

col = sns.color_palette("colorblind", 6)
# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, sharex=True, figsize=(10, 6))

# Generate the new log-spaced x-axis

colors1 = [col[1],col[2],col[0]]
labels1 = ['nDGP','f(R)','GR']
markers1 = ['o','D','P']

colors2 = [col[3],col[4],col[5]]
labels2 = ['PC1','PC2','PC3']
markers2 = ['x','.','^']
l_wl_max, l_gc_max = MGL_mu_lin.Survey.ells_wl_max, MGL_mu_lin.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL_mu_lin.Survey.l_wl, MGL_mu_lin.Survey.l_gc, MGL_mu_lin.Survey.l_xc

#print(MGL_mu_lin.Survey.mask_ells_wl.shape,MGL_mu_lin.Survey.mask_ells_gc.shape, MGL_mu_lin.Survey.mask_ells_xc.shape)


# Interpolate and plot on each subplot
for i in range(2):

    # Scatter points
    axs[0].scatter(l_wl[:ells_wl_bins[0]], Delta[i][:ells_wl_bins[0]], color=colors1[i], label=labels1[i], marker=markers1[i])

axs[0].set_title('Shear', fontsize=18)

for i in range(2):

    # Scatter points
    axs[1].scatter(l_wl[:ells_wl_bins[0]], -Usvd[i][:ells_wl_bins[0]], color=colors2[i], label=labels2[i], marker=markers2[i])

# Plot horizontal lines at y=0
axs[1].axhline(y=0, linestyle="--", color="k")

# Add labels and set x-axis limits and scale
axs[1].set_xlabel(r'$\ell$', fontsize=18)


axs[0].legend(fontsize=12, frameon=False)
axs[1].legend(fontsize=12, frameon=False)


axs[0].set_ylabel(r'$\Delta \mathbf{M}_{\text{ch}}$', fontsize=18)
axs[1].set_ylabel('Principal Components' + '\n' + r'(columns of $\mathbf{U}_{\text{ch}}$)', fontsize=18)

# Add x-axis label shared across all subplots
for ax in axs.flat:
    ax.set_xscale("log")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("Figures/Tests/PCA_components.png", dpi=300)

# Display the plot
plt.show()
