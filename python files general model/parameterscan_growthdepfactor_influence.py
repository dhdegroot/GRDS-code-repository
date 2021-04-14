from itertools import product

from kl_paretofront_support import *

"""
In this script we will compare a sensing genotype with a bet-hedging and an adaptive bet hedging cell. Assumptions are:  
- sensing cells experience a cost to the growth rate denoted by c, dependent on the number of environments c(N_e)
- there is a maximal switching rate for sensing cells (this mimicks a maximal adaptation speed)
- for bet-hedging cells the sum of switching from a phenotype cannot be larger than this maximal switching rate
- bet-hedging cells also have a minimal switching rate
- we assume that in each environment, there is only one phenotype with a fast growth rate, mu 

We vary:
- mean_durations, mean duration of an environment
- mu_max, growth rate of the good phenotype (rest has growth rate zero)
- num_phen, number of phenotypes
- num_env, number of environments
- sp_sensing_cost, cost of sensing = sp_sensing_cost * num_env

We optimise only in the case of bet-hedging. Then one optimal switching rate is chosen 

We end up with average growth rates for the chosen parameters. We can in the end scan some parameters and plot how
the growth rates change

"""
workingdir = os.getcwd()
np.random.seed(42)

# Options
RANDOM_TIMES = False
PLOT_DETAILS = True
MANY_PARS = True
SIM_TIME_FACTOR = 1
READ_IN_DATA = False
kinds = ['lin']
kinds_dict = {'const': 'constant', 'lin': 'linear', 'sensing': 'sensing'}

# Set parameter ranges
if MANY_PARS:
    num_envs = np.unique(np.floor(np.exp(np.linspace(np.log(4), np.log(20), 5))).astype(int))
    num_phen_factors = np.exp(np.linspace(np.log(.5), np.log(2), 3))
    mean_durations = np.linspace(5, 40, 5)
    mu_maxs = np.linspace(0.9, 1.2, 1)
    spread_dom_mu_factors = np.linspace(0.05, .9, 5)
    deltas = np.linspace(0.05, .25, 5)
else:
    num_envs = np.array([20])
    num_phen_factors = np.array([1.])
    mean_durations = np.array([50,100,200])
    mu_maxs = np.array([0.9])
    spread_dom_mu_factors = np.array([0.])
    deltas = np.array([1.0])

time_lb = 1000
sp_sensing_cost = 0.01
maxadaptationrate = 1
num_growth_dep_factors = 10
growth_dep_factors = np.exp(np.linspace(np.log(1), np.log(100), num_growth_dep_factors))

iterables = product(num_envs, num_phen_factors, mean_durations, mu_maxs, spread_dom_mu_factors, deltas)
n_simus = len(num_envs) * len(num_phen_factors) * len(mean_durations) * len(mu_maxs) * len(
    spread_dom_mu_factors) * len(deltas)

simulation = -1

results_df = pd.DataFrame(
    columns=['envnumber', 'lag', 'growthcost', 'lag_gen', 'meanmu', 'frac_max_mu', 'switchmin', 'switchmax', 'beta',
             'musens', 'dependency', 'meandelay', 'meandelay_gen', 'meangrowthcost', 'simulation', 'growth_dep_factor'])
mean_df = pd.DataFrame(
    columns=['dependency', 'meandelay', 'meandelay_gen', 'meangrowthcost', 'meanmu', 'frac_max_mu',
             'num_env', 'num_phen', 'mean_duration', 'mu_max', ' spread_dom_mu_factor',
             'delta', 'simulation', 'growth_dep_factor'])

for iterable in iterables:
    num_env, num_phen_factor, mean_duration, mu_max, spread_dom_mu_factor, delta = iterable
    num_phen = np.ceil(num_env * num_phen_factor).astype(int)
    if num_phen < 2:
        continue

    np.random.seed(1337)
    simulation += 1
    if (simulation + 1) % 10 == 0:
        print("Finished simulation " + str(simulation + 1) + " out of " + str(n_simus))

    sensing_cost = sp_sensing_cost * num_env

    """Create fitness landscapes, either sparse or dense"""
    # Create fitness matrix for each environment
    fit_mat = np.zeros((num_env, num_phen, num_phen))
    good_mus = np.random.uniform(mu_max * (1 - spread_dom_mu_factor), mu_max, num_env)

    for ind_env in range(num_env):
        good_mu = good_mus[ind_env]
        bad_mus = np.random.uniform(0, good_mu * (1 - delta), num_phen)
        ind_good_phen = ind_env % num_phen
        np.fill_diagonal(fit_mat[ind_env], bad_mus)
        fit_mat[ind_env, ind_good_phen, ind_good_phen] = good_mu

    """Create rest of system"""
    x0 = np.ones(num_phen) / num_phen

    # Make markov transition matrix
    # Either random
    trans_mat = np.random.rand(num_env, num_env)
    # Or just all equal
    # trans_mat = np.ones((num_env, num_env))
    np.fill_diagonal(trans_mat, 0)
    trans_mat = trans_mat / np.tile(np.sum(trans_mat, axis=0), (num_env, 1))

    # Choose environment durations
    # Either random:
    # avg_env_length = np.maximum(0, np.random.normal(mean_duration, 0.2 * mean_duration, num_env))
    # Or just all equal:
    avg_env_length = np.ones(num_env) * mean_duration

    # Choose simulation time long enough to touch upon a reasonable set of environment transitions
    min_sim_time = np.maximum(SIM_TIME_FACTOR * np.mean(avg_env_length) * num_env ** 2, 400)

    # Generate environment sequences and times
    env_seq = generate_env_seq(trans_mat, min_sim_time, avg_env_length, seed=1337,
                               random_times=RANDOM_TIMES)
    total_time = sum(env_seq[1])
    mean_mu_max = sum(good_mus[env_seq[0]] * env_seq[1]) / total_time
    mumin = np.min(fit_mat)
    mumax = np.max(fit_mat)

    # Gather some useful information
    # mumaxs = np.asarray([np.max(fitness_mat) for fitness_mat in fit_mat])
    # mumax = np.max(mumaxs)  # Fastest growth possible
    # mean_mu_max = sum(mumaxs[env_seq[0]] * env_seq[1]) / total_time  # Average growth Darwinian devil
    # mumin = np.min(fit_mat)  # Slowest growth rate possible (or zero)

    """Determine bounds for the switch rates"""
    switchratemax = maxadaptationrate / (num_phen - 1)
    switchratemin = switchratemax / 1000
    switchrate0 = switchratemin

    """Choose random switch rates for constant switching. Between each pair of phenotypes we take a random switch rate, 
    independent of the environment."""
    np.random.seed(42)
    switches_const_log = np.random.uniform(np.log(switchratemin), np.log(switchratemax), size=(num_phen, num_phen))

    for ind_growth_dep_factor, growth_dep_factor in enumerate(growth_dep_factors):
        """Choose switch rates for adaptive switching by picking from a range around the constant rates"""
        minimal_switches_adapt_log = switches_const_log - 0.5 * np.log(growth_dep_factor)
        maximal_switches_adapt_log = switches_const_log + 0.5 * np.log(growth_dep_factor)
        minimal_growth_per_phenotype = np.diag(np.min(fit_mat, axis=0))
        maximal_growth_per_phenotype = np.diag(np.max(fit_mat, axis=0))
        switches_adapt_log = np.zeros((num_env, num_phen, num_phen))
        for ind_env in range(num_env):
            rel_growth_rates = (np.diag(fit_mat[ind_env]) - minimal_growth_per_phenotype) / (
                    maximal_growth_per_phenotype - minimal_growth_per_phenotype)
            rel_noises_added = np.tile(1 - rel_growth_rates, (num_phen, 1))
            switches_adapt_log[ind_env] = minimal_switches_adapt_log + rel_noises_added * np.log(growth_dep_factor)

        """Analyse results. Calculate quantities capturing:
        - average adaptation time
        - average deviation from maximal growth rate
        """
        for kind in kinds:
            if (ind_growth_dep_factor != 0) & (kind not in ['lin']):
                continue
            dep = kinds_dict[kind]
            mu_dep = (kind not in ['const'])
            if kind is 'sensing':
                switch_rate = maxadaptationrate
                switch_matrices = None
            elif kind is 'const':
                switch_rate = None
                switch_matrices = np.tile(np.exp(switches_const_log), (num_env, 1, 1))
                for ind_env in range(num_env):
                    np.fill_diagonal(switch_matrices[ind_env],
                                     np.diag(switch_matrices[ind_env]) - np.sum(switch_matrices[ind_env], axis=0))
            elif kind is 'lin':
                switch_rate = None
                switch_matrices = np.exp(switches_adapt_log)
                for ind_env in range(num_env):
                    np.fill_diagonal(switch_matrices[ind_env],
                                     np.diag(switch_matrices[ind_env]) - np.sum(switch_matrices[ind_env], axis=0))

            noiserange, beta, musens, switchmax = (np.nan, np.nan, np.nan, np.nan)

            dummy, results_df_kind = kl_simulation(fit_mat, x0, num_env, num_phen, env_seq,
                                                   total_time, mean_mu_max, switch_rate, mu_dep=mu_dep,
                                                   dependency=dep, sensing_cost=sensing_cost,
                                                   switch_matrices=switch_matrices,
                                                   noiserange=noiserange, beta=beta, musens=musens, mumax=mumax)

            results_df_kind['switchmin'] = switch_rate
            results_df_kind['switchmax'] = switchmax
            results_df_kind['beta'] = beta
            results_df_kind['musens'] = musens
            results_df_kind['dependency'] = dep
            meandelay = np.mean(results_df_kind['lag'])
            results_df_kind['meandelay'] = meandelay
            meandelay_gen = np.mean(results_df_kind['lag_gen'])
            results_df_kind['meandelay_gen'] = meandelay_gen
            meangrowthcost = np.mean(results_df_kind['growthcost'])
            results_df_kind['meangrowthcost'] = meangrowthcost
            results_df_kind['simulation'] = simulation
            results_df_kind['growth_dep_factor'] = growth_dep_factor

            #mean_duration, mu_max, spread_dom_mu_factor, delta
            mean_df = mean_df.append(
                {'dependency': dep, 'meandelay': meandelay, 'meandelay_gen': meandelay_gen,
                 'meangrowthcost': meangrowthcost, 'meanmu': results_df_kind['meanmu'][0],
                 'frac_max_mu': results_df_kind['frac_max_mu'][0], 'num_env': num_env, 'num_phen': num_phen,
                 'mean_duration': mean_duration, 'mu_max': mu_max, ' spread_dom_mu_factor': spread_dom_mu_factor,
                 'delta': delta,
                 'simulation': simulation, 'growth_dep_factor': growth_dep_factor},
                ignore_index=True)
            results_df = results_df.append(results_df_kind)

"""Store results"""
results_df.to_csv(os.path.join(workingdir, "results", "kl_growthdepfactor_influence.csv"), index=False,
                  header=True)
mean_df.to_csv(os.path.join(workingdir, "results", "kl_growthdepfactor_influence_means.csv"), index=False,
               header=True)

"""Plot growthdepfactor versus ratio of growth rate versus initial growth rate"""
simus_list = mean_df['simulation'].unique()
figure, axes = plt.subplots(nrows=1, ncols=1)
# for ind_env in range(len(num_envs)):
#     for ind_duration in range(len(mean_durations)):
#         axes[ind_env, ind_duration].get_xaxis().set_visible(False)
#         axes[ind_env, ind_duration].get_yaxis().set_visible(False)

blackish = '#%02x%02x%02x' % (35, 31, 32)
simus_list = mean_df['simulation'].unique()
adaptive_df = mean_df[mean_df['dependency'] == 'linear']
no_advantage_simulations = []
for simu_ind, simulation in enumerate(simus_list):
    if simu_ind % 100 == 0:
        print("Plotting growth rate increase " + str(simu_ind) + " of " + str(n_simus))
    simu_df = adaptive_df[adaptive_df['simulation'] == simulation]
    base_growth = simu_df[simu_df['growth_dep_factor'] == 1.]['meanmu'].iloc[0]
    x = simu_df['growth_dep_factor']
    y = simu_df['meanmu'].values - base_growth
    if y[-1] <= 0:
        no_advantage_simulations.append(simulation)
    # ind_env = [ind_env for ind_env, num_env in enumerate(num_envs) if abs(num_env - simu_df['num_env'].values[0]) < 1e-6][0]
    # ind_duration = [ind for ind, duration in enumerate(mean_durations) if abs(duration - simu_df['mean_duration'].values[0]) < 1e-5][0]
    axes.plot(x, y, color=blackish, lw=0.10, zorder=-1, alpha=0.5)

x_avg = growth_dep_factors
y_avg = []
base_growths = adaptive_df[adaptive_df['growth_dep_factor'] == 1.]['meanmu']
for factor in growth_dep_factors:
    y_avg.append(np.mean(adaptive_df[adaptive_df['growth_dep_factor'] == factor]['meanmu'].values - base_growths))

axes.plot(x_avg, y_avg, color=blackish, lw=2, zorder=1)
axes.set_xlabel('growth rate dependence')
axes.set_ylabel('growth rate advantage')

plt.savefig(os.path.join(workingdir, "results", "growthdepfactor_influence_subtract.png"))
plt.savefig(os.path.join(workingdir, "results", "growthdepfactor_influence_subtract.svg"))

figure, axes = plt.subplots(nrows=1, ncols=1)
blackish = '#%02x%02x%02x' % (35, 31, 32)
simus_list = mean_df['simulation'].unique()
adaptive_df = mean_df[mean_df['dependency'] == 'linear']
for simu_ind, simulation in enumerate(simus_list):
    if simu_ind % 100 == 0:
        print("Plotting growth rate increase " + str(simu_ind) + " of " + str(n_simus))
    simu_df = adaptive_df[adaptive_df['simulation'] == simulation]
    base_growth = simu_df[simu_df['growth_dep_factor'] == 1.]['meanmu'].iloc[0]
    x = simu_df['growth_dep_factor']
    y = simu_df['meanmu'].values / base_growth
    # ind_env = [ind_env for ind_env, num_env in enumerate(num_envs) if abs(num_env - simu_df['num_env'].values[0]) < 1e-6][0]
    # ind_duration = [ind for ind, duration in enumerate(mean_durations) if abs(duration - simu_df['mean_duration'].values[0]) < 1e-5][0]
    axes.plot(x, y, color=blackish, lw=0.10, zorder=-1, alpha=0.5)

x_avg = growth_dep_factors
y_avg = []
base_growths = adaptive_df[adaptive_df['growth_dep_factor'] == 1.]['meanmu']
for factor in growth_dep_factors:
    y_avg.append(np.mean(adaptive_df[adaptive_df['growth_dep_factor'] == factor]['meanmu'].values / base_growths))

axes.plot(x_avg, y_avg, color=blackish, lw=2, zorder=1)
axes.set_xlabel('growth rate dependence')
axes.set_ylabel('growth rate advantage')

plt.savefig(os.path.join(workingdir, "results", "growthdepfactor_influence.png"))
plt.savefig(os.path.join(workingdir, "results", "growthdepfactor_influence.svg"))

# figure, axes = plt.subplots(nrows=1, ncols=1)
# # for ind_env in range(len(num_envs)):
# #     for ind_duration in range(len(mean_durations)):
# #         axes[ind_env, ind_duration].get_xaxis().set_visible(False)
# #         axes[ind_env, ind_duration].get_yaxis().set_visible(False)
#
# blackish = '#%02x%02x%02x' % (35, 31, 32)
# simus_list = mean_df['simulation'].unique()
# adaptive_df = mean_df[mean_df['dependency'] == 'linear']
# for simu_ind, simulation in enumerate(simus_list):
#     if simu_ind % 100 == 0:
#         print("Plotting paretofront " + str(simu_ind) + " of " + str(n_simus))
#     simu_df = adaptive_df[adaptive_df['simulation'] == simulation]
#     base_growth = simu_df[simu_df['growth_dep_factor'] == 1.]['meanmu'].iloc[0]
#     x = simu_df['growth_dep_factor']
#     y = simu_df['meanmu'].values - base_growth
#     # ind_env = [ind_env for ind_env, num_env in enumerate(num_envs) if abs(num_env - simu_df['num_env'].values[0]) < 1e-6][0]
#     # ind_duration = [ind for ind, duration in enumerate(mean_durations) if abs(duration - simu_df['mean_duration'].values[0]) < 1e-5][0]
#     axes.plot(x, y, color=blackish, lw=0.10, zorder=-1, alpha=0.5)
#
# x_avg = growth_dep_factors
# y_avg = []
# base_growths = adaptive_df[adaptive_df['growth_dep_factor'] == 1.]['meanmu']
# for factor in growth_dep_factors:
#     y_avg.append(np.mean(adaptive_df[adaptive_df['growth_dep_factor'] == factor]['meanmu'].values - base_growths))
#
# y_predicted = np.log(0.5+0.5*x_avg)/100
#
# axes.plot(x_avg, y_avg, color=blackish, lw=2, zorder=1)
# axes.plot(x_avg, y_predicted, color='red', lw=2, zorder=2)
# axes.set_xlabel('growth rate dependence')
# axes.set_ylabel('growth rate advantage')

print(*no_advantage_simulations, sep=',')
plt.show()
