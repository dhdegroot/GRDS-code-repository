from itertools import product

from kl_paretofront_support import *
from matplotlib import cm

"""
In this script we will study the effect of growth rate dependent switching rates for many different parameter sets.
The sampling of the parameters is described in the SI of the paper "Coupling phenotype stability to growth rate 
overcomes limitations of bet-hedging strategies"

"""
workingdir = os.getcwd()
np.random.seed(42)

# Options
RANDOM_TIMES = True
# PLOT_DETAILS = True
MANY_PARS = True
SIM_TIME_FACTOR = 1
READ_IN_DATA = False
kinds = ['lin']
kinds_dict = {'const': 'constant', 'lin': 'linear', 'sensing': 'sensing'}
custom_colors = cm.get_cmap('tab10')

# Set parameter ranges
if MANY_PARS:
    num_envs = np.unique(np.floor(np.exp(np.linspace(np.log(4), np.log(200), 10))).astype(int))
    num_phen_factors = [1.]
    mu_maxs = np.linspace(0.9, 1.2, 1)
    spread_dom_mu_factors = np.linspace(0.05, 0.65, 5)
    deltas = np.linspace(0.05, .25, 5)

    # num_envs = np.unique(np.floor(np.exp(np.linspace(np.log(4), np.log(20), 5))).astype(int))
    # mu_maxs = np.linspace(0.9, 1.2, 1)
    # spread_dom_mu_factors = np.linspace(0.05, 0.45, 5)
    # deltas = np.linspace(0.05, .45, 5)

    n_reps = 20
    reps = np.arange(n_reps)

time_lb = 1000
maxadaptationrate = 1

iterables = product(num_envs, num_phen_factors, reps, mu_maxs, spread_dom_mu_factors, deltas)
n_simus = len(num_envs) * len(num_phen_factors) * len(reps) * len(mu_maxs) * len(
    spread_dom_mu_factors) * len(deltas)

simulation = -1

results_df = pd.DataFrame(
    columns=['pred_GRDS_benefit', 'num_env', 'num_phen', 'mu_max', 'spread_dom_mu_factor',
             'delta', 'simulation'])

np.random.seed(1337)
seeds = np.random.randint(0, 1000000, n_simus)
if not READ_IN_DATA:
    for iterable in iterables:
        num_env, num_phen_factor, rep, mu_max, spread_dom_mu_factor, delta = iterable
        num_phen = np.ceil(num_env * num_phen_factor).astype(int)
        if num_phen < 2:
            continue

        np.random.seed(seeds[simulation])
        simulation += 1
        if (simulation + 1) % 10 == 0:
            print("Finished simulation " + str(simulation + 1) + " out of " + str(n_simus))

        """Create fitness landscapes, either sparse or dense"""
        # Create fitness matrix for each environment
        fit_mat = np.zeros((num_env, num_phen, num_phen))
        good_mus = np.random.uniform(mu_max * (1 - spread_dom_mu_factor), mu_max, num_env)

        for ind_env in range(num_env):
            good_mu = good_mus[ind_env]
            bad_mus = np.random.uniform(0, good_mu * (1 - delta), num_phen)
            # bad_mus = np.random.uniform(good_mu * (1 - delta), good_mu, num_phen)
            ind_good_phen = ind_env % num_phen
            np.fill_diagonal(fit_mat[ind_env], bad_mus)
            fit_mat[ind_env, ind_good_phen, ind_good_phen] = good_mu

        # Make markov transition matrix
        trans_mat = np.random.rand(num_env, num_env)
        np.fill_diagonal(trans_mat, 0)
        trans_mat = trans_mat / np.tile(np.sum(trans_mat, axis=0), (num_env, 1))

        # fit_mat = np.zeros((2, 2, 2))
        # fit_mat[0, 0, 0] = 0.
        # fit_mat[0, 1, 1] = -1.
        # fit_mat[1, 0, 0] = 0.6
        # fit_mat[1, 1, 1] = 0.8
        # trans_mat = np.array([[0., 1], [1, 0]])
        # num_env = 2

        # Get probabilities of environments as right eigenvector with eigenvalue 1
        eig_val_b, eig_vec_b = np.linalg.eig(trans_mat)
        eig_vec_p = eig_vec_b[:, np.isclose(1, eig_val_b.real)].real
        p = (eig_vec_p / np.sum(eig_vec_p)).flatten()

        # Choose environment durations
        pred_GRDS_benefit = 0
        for ind_env_j in range(num_env):
            ind_good_phen_j = ind_env_j % num_phen
            mu_j_j = fit_mat[ind_env_j, ind_good_phen_j, ind_good_phen_j]
            for ind_env_i in range(ind_env_j + 1, num_env):
                ind_good_phen_i = ind_env_i % num_phen
                mu_i_i = fit_mat[ind_env_i, ind_good_phen_i, ind_good_phen_i]
                mu_j_i = fit_mat[ind_env_j, ind_good_phen_i, ind_good_phen_i]
                mu_i_j = fit_mat[ind_env_i, ind_good_phen_j, ind_good_phen_j]
                fij = (mu_j_j - mu_j_i) / (mu_j_j - mu_j_i + mu_i_i - mu_i_j)

                ij_term = p[ind_env_j] * trans_mat[ind_env_i, ind_env_j] * fij * (mu_j_j - mu_i_j)
                ji_term = p[ind_env_i] * trans_mat[ind_env_j, ind_env_i] * (1 - fij) * (mu_i_i - mu_j_i)

                pred_GRDS_benefit += (ij_term + ji_term)
                # if (ij_term + ji_term < 0):
                #     print('ji_term, ij_term, simulation, ind_env_j, ind_env_i')
                #     print(ji_term, ij_term, simulation, ind_env_j, ind_env_i)

        results_df = results_df.append(
            {'pred_GRDS_benefit': pred_GRDS_benefit, 'num_env': num_env, 'num_phen': num_phen, 'mu_max': mu_max,
             'spread_dom_mu_factor': spread_dom_mu_factor,
             'delta': delta, 'simulation': simulation},
            ignore_index=True)

fig, ax = plt.subplots(figsize=(15, 6.5))
im = ax.scatter(results_df['num_env'], results_df['pred_GRDS_benefit'],
                c=results_df['spread_dom_mu_factor'] / results_df['delta'])
# im = ax.scatter(results_df['num_env'], results_df['pred_GRDS_benefit'])
ax.set_xlabel('Number of environments')
ax.set_ylabel('Predicted GRDS benefit')
ax.set_ylim(0)
fig.colorbar(im, ax=ax, label='Spread dominant growth rates/Spectral gap')

fig.savefig(os.path.join(workingdir, 'results', 'pred_GRDS_benefit.png'))

plt.show()
