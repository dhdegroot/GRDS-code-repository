from kl_optimisation_support import *

"""This script is intended to check how well we analytically approximate the eigenvectors and eigenvalues."""
# Set parameter values
num_env = 5
num_phen = 10
iterations = 1  # number of iterations within each calculation of deviation from max mean growth rate
min_sim_time = 10 ** 4
avg_env_length = 50

max_mu = 1.5
min_mu = -0.1
seed = 42

switch_rate = 0.006
max_switch_rate = 0.01
beta = 0.45

# Initialise components of fitness matrix
growth_rates = np.linspace(max_mu, min_mu, num_phen)
fit_mat = np.zeros((num_env, num_phen, num_phen))

# Create transition probability matrix, switching to the same environment is not allowed,
# switching to others is equally likely everywhere
trans_mat = np.ones((num_env, num_env)) * (1 / (num_env - 1))
np.fill_diagonal(trans_mat, 0)

# Fill diagonal of fitness matrices
for ind in range(num_env):
    np.fill_diagonal(fit_mat[ind], np.concatenate((growth_rates[-ind:], growth_rates[:-ind])))

a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=True,
                      max_switch_rate=max_switch_rate, beta=beta)
# TODO: check how close eigenvalues are
eig_vals, eig_vecs = np.linalg.eig(a_mat)
m_inverse = np.linalg.inv(eig_vecs)

zero_vals = np.zeros((num_env, num_phen))
first_vals = np.zeros((num_env, num_phen))
second_vals = np.zeros((num_env, num_phen))
zero_vecs = np.zeros((num_env, num_phen, num_phen))
first_vecs = np.zeros((num_env, num_phen, num_phen))
second_vecs = np.zeros((num_env, num_phen, num_phen))
zero_inv = np.zeros((num_env, num_phen, num_phen))
first_inv = np.zeros((num_env, num_phen, num_phen))
second_inv = np.zeros((num_env, num_phen, num_phen))
for i in range(num_env):
    zero_vals[i], first_vals[i], second_vals[i], zero_vecs[i], first_vecs[i], second_vecs[i], zero_inv[i], first_inv[i], \
    second_inv[i] = calc_eig_vals_vecs_approximations(a_mat[i], fit_mat[i])

"""We calculate the fraction of the cells in dominant ev of env j that is projected to dominant ev of env i"""
# We do it up to first order here
# First look up which eigenvector is dominant in the different environments
dominant_eigs_first = [np.argmax(row) for row in first_vals]
q_ij_first = calc_q_ij(a_mat, fit_mat, dominant_eigs_first, order='first')

# Test if this coincides with if we calculate it from the eigenvectors and inverses
q_ij_first_mminv = np.zeros((num_env, num_env))
for env_i in range(num_env):
    alpha_i = dominant_eigs_first[env_i]
    for env_j in range(num_env):
        alpha_j = dominant_eigs_first[env_j]
        q_ij_first_mminv[env_i,env_j] = np.matmul(first_inv[env_i], first_vecs[env_j])[alpha_i,alpha_j]

# First look up which eigenvector is dominant in the different environments
dominant_eigs_second = [np.argmax(row) for row in second_vals]
q_ij_second = calc_q_ij(a_mat, fit_mat, dominant_eigs_second, order='second')

# Calculate average fitness
avg_fitness = calc_approx_fitness(a_mat, fit_mat, avg_env_length, trans_mat, order='second')

# Test these q_ij calculations
q_ij_test_first, q_ij_test_second = test_q_ij(a_mat, fit_mat, dominant_eigs_first, zero_vecs, zero_inv, first_vecs, first_inv, second_vecs, second_inv)

# Test if this coincides with if we calculate it from the eigenvectors and inverses
q_ij_second_mminv = np.zeros((num_env, num_env))
for env_i in range(num_env):
    alpha_i = dominant_eigs_second[env_i]
    for env_j in range(num_env):
        alpha_j = dominant_eigs_second[env_j]
        minv_m = np.matmul(second_inv[env_i], second_vecs[env_j])
        q_ij_second_mminv[env_i, env_j] = minv_m[alpha_i,alpha_j]

"""After this, we sort and normalize the calculated eigenvectors, so that we can compare them"""
# Sort and normalize eigenvalues and corresponding eigenvectors
processed_eig_vecs = np.zeros(np.shape(eig_vecs))
for i in range(num_env):
    sorted_eigs_inds = eig_vals[i, :].argsort()[::-1]
    eig_vals[i] = eig_vals[i, sorted_eigs_inds]
    for j in range(num_phen):
        eig_vec = eig_vecs[i, :, sorted_eigs_inds[j]]
        processed_eig_vecs[i, :, j] = eig_vec / (np.sign(np.sum(eig_vec)) * np.linalg.norm(eig_vec))

# Sort and normalize eigenvalues and corresponding eigenvectors
processed_zero_vecs = np.zeros(np.shape(zero_vecs))
processed_zero_inv = np.zeros(np.shape(zero_inv))
for i in range(num_env):
    sorted_eigs_inds = zero_vals[i, :].argsort()[::-1]
    zero_vals[i] = zero_vals[i, sorted_eigs_inds]
    for j in range(num_phen):
        eig_vec = zero_vecs[i, :, sorted_eigs_inds[j]]
        processed_zero_vecs[i, :, j] = eig_vec / (np.sign(np.sum(eig_vec)) * np.linalg.norm(eig_vec))
        # Exchange rows in inverse, like you exchange eigenvectors
        inv_vec = zero_inv[i, sorted_eigs_inds[j], :]
        processed_zero_inv[i, j, :] = inv_vec / (np.sign(np.sum(inv_vec)) * np.linalg.norm(inv_vec))

# Sort and normalize eigenvalues and corresponding eigenvectors
processed_first_vecs = np.zeros(np.shape(first_vecs))
processed_first_inv = np.zeros(np.shape(zero_inv))
for i in range(num_env):
    sorted_eigs_inds = first_vals[i, :].argsort()[::-1]
    first_vals[i] = first_vals[i, sorted_eigs_inds]
    for j in range(num_phen):
        eig_vec = first_vecs[i, :, sorted_eigs_inds[j]]
        processed_first_vecs[i, :, j] = eig_vec / (np.sign(np.sum(eig_vec)) * np.linalg.norm(eig_vec))
        # Exchange rows in inverse, like you exchange eigenvectors
        inv_vec = first_inv[i, sorted_eigs_inds[j], :]
        processed_first_inv[i, j, :] = inv_vec / (np.sign(np.sum(inv_vec)) * np.linalg.norm(inv_vec))

# Sort and normalize eigenvalues and corresponding eigenvectors
processed_second_vecs = np.zeros(np.shape(second_vecs))
processed_second_inv = np.zeros(np.shape(zero_inv))
for i in range(num_env):
    sorted_eigs_inds = second_vals[i, :].argsort()[::-1]
    second_vals[i] = second_vals[i, sorted_eigs_inds]
    for j in range(num_phen):
        eig_vec = second_vecs[i, :, sorted_eigs_inds[j]]
        processed_second_vecs[i, :, j] = eig_vec / (np.sign(np.sum(eig_vec)) * np.linalg.norm(eig_vec))
        # Exchange rows in inverse, like you exchange eigenvectors
        inv_vec = second_inv[i, sorted_eigs_inds[j], :]
        processed_second_inv[i, j, :] = inv_vec / (np.sign(np.sum(inv_vec)) * np.linalg.norm(inv_vec))

diff_zero_order = [np.sum(np.abs(eig_vals - zero_vals))/(eig_vals.size), np.sum(np.abs(processed_eig_vecs - processed_zero_vecs))/(eig_vecs.size)]
diff_first_order = [np.sum(np.abs(eig_vals - first_vals))/(eig_vals.size), np.sum(np.abs(processed_eig_vecs - processed_first_vecs))/(eig_vecs.size)]
diff_second_order = [np.sum(np.abs(eig_vals - second_vals))/(eig_vals.size), np.sum(np.abs(processed_eig_vecs - processed_second_vecs))/(eig_vecs.size)]
diff_first_second = processed_first_vecs-processed_second_vecs
print('Avg difference per entry between the approximation and the actual eigen vectors was:')
print('for eigenvalues, and eigenvectors, respectively')
print('Zeroth order approximation:')
print(','.join(map(str, diff_zero_order)))
print('First order approximation:')
print(','.join(map(str, diff_first_order)))
print('Second order approximation:')
print(','.join(map(str, diff_second_order)))
