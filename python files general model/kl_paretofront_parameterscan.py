import os
from itertools import product

import seaborn as sns

from kl_paretofront_support import *

"""In this script we will sample many different parameter sets that determine a growing system in a fluctuating 
environment. We will focus on systems with a sparse fitness-landscape. Each environment will only contain one very
dominant growth rate, and many non-dominant ones. To simplify matters we will assume that # environments = # phenotypes.
Phenotype i is the dominant one in environment i. We sample:
- 1 dominant growth rate from a uniform distribution on the interval [mumin,mumax]
- n-1 non-dominant growth rates, s.t. mu_dom - mu_non in [deltamin,deltamax], with deltamin large enough
- order of environments (Markov chain, equal weights)
- environment time of specific instance of environment (from exponential distribution)

The following we just take fixed
- Average environment lengths will just be taken equal
- Number of phenotypes n can be the same
- noiserange
- musens

The procedure will be as follows
1) Loop over num_iter sets of parameters
2) Loop over num_switch sets of H0's. This determines the switching rate in the non-mudependent case
3) Loop over num_betas. Beta will determine behaviour of switching via sigmoid, according to formula
            H = noiserange * H0 (mumax-mu)^beta/((mumax-musens)^beta+(mumax-mu)^beta) + H0
        where noiserange is the maximal fold-change that a cell can add to the noise,
        and musens is the inflection point of the sigmoid.
4) Gather all information in parameterset-dataframe

5) Do simulations. For each environment, store delay-time and difference stationary growth rate from dominant growth 
        rate

6) Calculate average delay-time, average difference dominant growth rate and stationary growth rate, 
        and average growth rate for each parameterset
7) Make scatterplot with x-axis the delay-time, y-axis the average growth rate cost, colour the average fitness 
"""
workingdir = os.getcwd()
np.random.seed(42)

results_df = pd.DataFrame(
    columns=['simulation', 'num_env', 'num_phen', 'mean_duration', 'total_time', 'mumax', 'mean_mu_max',
             'spread_dom_mu_factor', 'delta', 'switchrate', 'G', 'growthcost', 'growthdelay'])

# Set parameter values
num_envs = np.unique(np.floor(np.exp(np.linspace(np.log(2), np.log(40), 10))).astype(int))
num_phen_factors = np.unique(np.floor(np.linspace(1, 2, 4)).astype(int))
mean_durations = np.linspace(5, 100, 10)
mu_maxs = np.linspace(0.9, 1.2, 1)
spread_dom_mu_factors = np.linspace(0.05, 1, 5)
deltas = np.linspace(0.05, .25, 5)
sim_time_factor = 0.1
time_lb = 1000
time_ub = 10000
num_switch = 40

READ_IN_DATA = False

iterables = product(num_envs, num_phen_factors, mean_durations, mu_maxs, spread_dom_mu_factors, deltas)
n_simus = len(num_envs) * len(num_phen_factors) * len(mean_durations) * len(mu_maxs) * len(
    spread_dom_mu_factors) * len(deltas)

if not READ_IN_DATA:
    simulation = -1
    for iterable in iterables:
        simulation += 1
        if (simulation + 1) % 2 == 0:
            print("Finished simulation " + str(simulation + 1) + " out of " + str(n_simus))

        num_env, num_phen_factor, mean_duration, mumax, spread_dom_mu_factor, delta = iterable
        num_phen = num_env * num_phen_factor

        # Choose initial value
        x0 = np.ones(num_phen) / num_phen
        # Make markov transition matrix
        trans_mat = np.random.rand(num_env, num_env)
        np.fill_diagonal(trans_mat, 0)
        trans_mat = trans_mat / np.tile(np.sum(trans_mat, axis=0), (num_env, 1))

        # Choose environment durations
        avg_env_length = np.maximum(0, np.random.normal(mean_duration, 0.2 * mean_duration, num_env))
        # Choose simulation time long enough to touch upon a reasonable set of environment transitions
        min_sim_time = np.minimum(np.maximum(sim_time_factor * np.mean(avg_env_length) * num_env ** 2, time_lb),
                                  time_ub)
        # Generate environment sequences and times
        env_seq = generate_env_seq(trans_mat, min_sim_time, avg_env_length, seed=np.random.randint(10000))
        total_time = sum(env_seq[1])
        fit_mat = np.zeros((num_env, num_phen, num_phen))
        good_mus = np.random.uniform(mumax * (1 - spread_dom_mu_factor), mumax, num_env)

        for ind_env in range(num_env):
            good_mu = good_mus[ind_env]
            bad_mus = np.random.uniform(0, good_mu * (1 - delta), num_phen)
            np.fill_diagonal(fit_mat[ind_env], bad_mus)
            fit_mat[ind_env, ind_env, ind_env] = good_mu

        mean_mu_max = sum(good_mus[env_seq[0]] * env_seq[1]) / total_time
        mumin = np.min(fit_mat)

        # Set some parameter-dependent optimisation parameters
        switchratemax = 1 / (2.5 * num_phen)
        switchratemin = switchratemax / 100

        # Choose switch rates and betas which should be tried
        switchrates = np.exp(np.linspace(np.log(switchratemin), np.log(switchratemax), num_switch))

        for switchrate in switchrates:
            dummy, results_df_simu = kl_simulation(fit_mat, x0, num_env, num_phen, env_seq,
                                                   total_time, mean_mu_max, switch_rate=switchrate, mu_dep=False)

            meanlag = np.mean(results_df_simu['lag'])
            meangrowthcost = np.mean(results_df_simu['growthcost'])
            results_df = results_df.append(
                {'simulation': simulation, 'num_env': num_env, 'num_phen': num_phen, 'mean_duration': mean_duration,
                 'total_time': total_time, 'mumax': mumax, 'mean_mu_max': mean_mu_max,
                 'spread_dom_mu_factor': spread_dom_mu_factor, 'delta': delta, 'switchrate': switchrate,
                 'G': results_df_simu['meanmu'][0], 'growthcost': meangrowthcost, 'growthdelay': meanlag},
                ignore_index=True)

    """---------------------------------------------------------------------------------------------------------"""
    # Store data
    results_df.to_csv(os.path.join(workingdir, "results", "kl_parameterscan_paretofront.csv"), index=False, header=True)
else:
    num_env_double = True
    if num_env_double:
        results_df = pd.read_csv(os.path.join(workingdir, "results", "kl_parameterscan_paretofront_doublenumenv2.csv"))
        # Get rid of double num_env = 2, if it is there (was there as an error in an old dataset)
        results_df = results_df.drop(results_df.index[40000:80000])
    else:
        results_df = pd.read_csv(os.path.join(workingdir, "results", "kl_parameterscan_paretofront.csv"))


custom_colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
blackish = '#%02x%02x%02x' % (35, 31, 32)

# Create dataframe with only the optima
optima_df = pd.DataFrame(columns=results_df.columns)

"""Make lineplot of x=mean lagtimes, y=mean growth cost (per simulation), indicate optimum per 
        pareto-front by point"""

figure, axes = plt.subplots(nrows=len(num_envs), ncols=len(mean_durations), figsize=(30, 30))
for ind_env in range(len(num_envs)):
    for ind_duration in range(len(mean_durations)):
        axes[ind_env, ind_duration].get_xaxis().set_visible(False)
        axes[ind_env, ind_duration].get_yaxis().set_visible(False)

for simu_ind, simulation in enumerate(np.unique(results_df['simulation'].values).astype(int)):
    if simu_ind % 100 == 0:
        print("Plotting paretofront " + str(simu_ind) + " of " + str(n_simus))
    simu_df = results_df[results_df['simulation'] == simulation]
    ind_env = [ind_env for ind_env, num_env in enumerate(num_envs) if abs(num_env - simu_df['num_env'].values[0]) < 1e-6][0]
    ind_duration = [ind for ind, duration in enumerate(mean_durations) if abs(duration - simu_df['mean_duration'].values[0]) < 1e-5][0]
    axes[ind_env, ind_duration].plot(simu_df['growthdelay'], simu_df['growthcost'], color=blackish, lw=0.10,
                                     zorder=-1, alpha=0.5)
    # optima_df = optima_df.append(simu_df.iloc[np.argmax(simu_df['G'].values), :])

# plot_optima = False
# if plot_optima:
#     for simulation in range(n_simus):
#         growthdelay_simu = optima_df[optima_df['simulation'] == simulation]['growthdelay'].values
#         growthcost_simu = optima_df[optima_df['simulation'] == simulation]['growthcost'].values
#         ax.scatter(x=growthdelay_simu, y=growthcost_simu, c=custom_colors[0], s=5, alpha=0.5,
#                    zorder=1)

# axes[0, 0].set_xlabel("mean growth delay")
# axes[0, 0].set_ylabel("mean growth rate cost")

plt.savefig(os.path.join(workingdir, "results", "kl_paretofront_parameterscan.png"))
plt.savefig(os.path.join(workingdir, "results", "kl_paretofront_parameterscan.svg"))

# g = sns.FacetGrid(data=results_df, col="num_env", row="mean_duration", hue='simulation',
#                   legend_out=True)  # , height=4, aspect=.5
# g.map(sns.lineplot, "growthdelay", "growthcost")
# # g.add_legend()
# plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_panel_overlap.png"))
# plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_panel_overlap.svg"))

plt.show()
