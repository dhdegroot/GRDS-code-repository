import os
from itertools import product

from scipy import optimize

from kl_paretofront_support import *

"""
In this script we will create one fitness landscape and try to optimize the population growth rate. We will optimize
a constant, linear, exponential and sigmoidal relation between growth rate and switching rates. Both will be limited by 
a maximal, and a minimal switching rate.   

Phenotype i is the dominant one in environment i. We sample:
- 1 dominant growth rate from a uniform distribution on the interval [mumin,mumax]
- n-1 non-dominant growth rates, s.t. mu_dom - mu_non in [deltamin,deltamax], with deltamin large enough
- order of environments (Markov chain, equal weights)
- environment time of specific instance of environment (from exponential distribution)

The following can be optimized
- switching rate

- beta
- noiserange
- minimal switching rate
- musens

The procedure will be as follows
1) Choose set of parameters that determines fitness landscape and env_seq
2) Use optimization function to determine optimal constant switching rate
3) Use optimization function to determine optimal sigmoid. This sigmoid will be given by 
        H = noiserange * H0*(mumax-mu)^beta/((mumax-musens)^beta+(mumax-mu)^beta) + H0
    where noiserange is the maximal fold-change that a cell can add to the noise,
    and musens is the inflection point of the sigmoid.
4) For optimal parameters: calculate time-averaged mu-distribution for constant case, and mu-dependent case
5) Plot these distributions
6) Plot sigmoid and constant noise as a function of mu

"""
workingdir = os.getcwd()
np.random.seed(42)

# Initialize dataframe in which all parameters and resulting growth rates will be stored
results_df = pd.DataFrame(
    columns=['simulation', 'num_env', 'num_phen', 'mean_duration', 'total_time', 'mumax', 'mean_mu_max',
             'spread_dom_mu', 'deltamax', 'switchratemin', 'switchratemax',
             'G_const', 'G_lin', 'G_exp', 'G_sigm', 'Gadv_lin', 'Gadv_exp', 'Gadv_sigm', 'const', 'exp_min', 'exp_max',
             'exp_beta', 'lin_min', 'lin_max',
             'sigm_min', 'sigm_max', 'sigm_beta', 'sigm_musens'])

# Set options
OPT_SIGMOID_SWITCHMAXMIN = True
# switch_rate_basis = 'general'
verbose_optimization = False
verbose_results = True
READ_IN_DATA = False
PLOT_SINGLE_SIMU = True
DEBUG_SINGLES = False

if not READ_IN_DATA:
    # Set parameter values for system
    num_envs = np.unique(np.floor(np.exp(np.linspace(np.log(2), np.log(20), 6))).astype(int))
    num_phen_factors = np.unique(np.floor(np.linspace(1, 2, 2)).astype(int))
    mean_durations = np.linspace(5, 200, 5)
    mu_maxs = np.linspace(0.9, 1.2, 1)
    spread_dom_mus = np.linspace(0.1, 0.45, 5)
    # deltas = np.linspace(0.05, .25, 3)
    deltamaxs = np.linspace(0.1, 0.45, 5)
    sim_time_factor = 0.5
    time_lb = 1000
    time_ub = 10000

    # Set optimisation parameters
    # beta
    beta0 = 1
    betamin = 0.0000001
    betamax = 100

    # switchrate
    switchrate0 = 0.01

    # maxswitchrate
    maxswitchrate0 = 0.1

    # iterables = product(num_envs, num_phen_factors, mean_durations, mu_maxs, spread_dom_mu_factors, deltas)
    # n_simus = len(num_envs) * len(num_phen_factors) * len(mean_durations) * len(mu_maxs) * len(
    #     spread_dom_mu_factors) * len(deltas)
    iterables = product(num_envs, num_phen_factors, mean_durations, mu_maxs, spread_dom_mus, deltamaxs)
    n_simus = len(num_envs) * len(num_phen_factors) * len(mean_durations) * len(mu_maxs) * len(
        spread_dom_mus) * len(deltamaxs)
    simulation = -1
    seeds = np.random.randint(0, n_simus ** 2, n_simus)
    for iterable in iterables:
        np.random.seed(seeds[simulation])
        simulation += 1
        if (simulation + 1) % 10 == 0:
            print("Finished simulation " + str(simulation + 1) + " out of " + str(n_simus))

        # num_env, num_phen_factor, mean_duration, mumax, spread_dom_mu_factor, delta = iterable
        # num_phen = num_env * num_phen_factor
        num_env, num_phen_factor, mean_duration, mumax, spread_dom_mu, deltamax = iterable
        num_phen = num_env * num_phen_factor

        # Choose initial value
        x0 = np.ones(num_phen) / num_phen
        # Make markov transition matrix
        # Either random
        # trans_mat = np.random.rand(num_env, num_env)
        # Or just all equal
        trans_mat = np.ones((num_env, num_env))
        np.fill_diagonal(trans_mat, 0)
        trans_mat = trans_mat / np.tile(np.sum(trans_mat, axis=0), (num_env, 1))

        # Choose environment durations
        # avg_env_length = np.maximum(0, np.random.normal(mean_duration, 0.2 * mean_duration, num_env))
        avg_env_length = np.ones(num_env)*mean_duration
        # Choose simulation time long enough to touch upon a reasonable set of environment transitions
        min_sim_time = np.minimum(np.maximum(sim_time_factor * np.mean(avg_env_length) * num_env ** 2, time_lb),
                                  time_ub)
        # Generate environment sequences and times
        env_seq = generate_env_seq(trans_mat, min_sim_time, avg_env_length, seed=np.random.randint(10000))
        total_time = sum(env_seq[1])
        fit_mat = np.zeros((num_env, num_phen, num_phen))
        # good_mus = np.random.uniform(mumax * (1 - spread_dom_mu_factor), mumax, num_env)
        good_mus = np.linspace(mumax - spread_dom_mu, mumax, num_env)

        for ind_env in range(num_env):
            good_mu = good_mus[ind_env]
            # bad_mus = np.random.uniform(0, good_mu * (1 - delta), num_phen)
            # if num_phen == 2:
            #     # bad_mus = [(good_mu * (1 - delta)) / 2]
            #     bad_mus = [good_mu - deltamax / 2]
            # else:
            #     # bad_mus = np.linspace(0, good_mu * (1 - delta), num_phen - 1)
            #    bad_mus = np.linspace(good_mu - deltamax, good_mu, num_phen)[:-1]
            all_mus = np.linspace(good_mu - deltamax, good_mu, num_phen)

            counter = 0
            np.fill_diagonal(fit_mat[ind_env],np.concatenate((all_mus[-(ind_env+1):],all_mus[:-(ind_env+1)])))
            # for ind_phen in range(num_phen):
            #     if ind_phen == ind_env:
            #         continue
            #     fit_mat[ind_env, ind_phen, ind_phen] = bad_mus[counter]
            #     counter += 1
            # # np.fill_diagonal(fit_mat[ind_env], bad_mus)
            # fit_mat[ind_env, ind_env, ind_env] = good_mu

        mean_mu_max = sum(good_mus[env_seq[0]] * env_seq[1]) / total_time
        mumin = np.min(fit_mat)

        # Set some parameter-dependent optimisation parameters
        switchratemax = 1 / (2.5 * num_phen)
        switchratemin = switchratemax / 10000
        maxswitchratemax = switchratemax
        maxswitchratemin = switchratemin
        musens0 = (mumax + mumin) / 2
        musensmin = mumin
        musensmax = mumax - 0.01

        """Define optimization functions"""


        def fun_const_switching(x):
            if verbose_optimization:
                print('switching rate: ' + str(np.exp(x)))
            frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max,
                                              switch_rate=np.exp(x),
                                              mu_dep=False)
            return 1 - frac_max_mu


        def fun_exp_switching(x):
            """x contains as entries: minswitchrate, maxswitchrate, beta, musens"""
            if verbose_optimization:
                print('min switching rate: ' + str(np.exp(x[0])) + ', max switching rate: ' + str(
                    np.exp(x[1])) + ', beta: ' + str(np.exp(x[2])))
            noiserange = (np.exp(x[1]) - np.exp(x[0])) / np.exp(x[0])
            frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq,
                                              total_time, mean_mu_max,
                                              switch_rate=np.exp(x[0]), mu_dep=True, noiserange=noiserange, mumin=mumin,
                                              beta=np.exp(x[2]), mumax=mumax, dependency='exponential')

            return 1 - frac_max_mu


        def fun_lin_switching(x):
            """x contains as entries: minswitchrate, maxswitchrate, beta, musens"""
            if verbose_optimization:
                print('min switching rate: ' + str(np.exp(x[0])) + ', max switching rate: ' + str(np.exp(x[1])))
            noiserange = (np.exp(x[1]) - np.exp(x[0])) / np.exp(x[0])
            frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq,
                                              total_time, mean_mu_max,
                                              switch_rate=np.exp(x[0]), mu_dep=True, noiserange=noiserange, mumin=mumin,
                                              mumax=mumax, dependency='linear')

            return 1 - frac_max_mu


        if OPT_SIGMOID_SWITCHMAXMIN:
            def fun_sigmoid_switching(x):
                """x contains as entries: minswitchrate, maxswitchrate, beta, musens"""
                if verbose_optimization:
                    print('min switching rate: ' + str(np.exp(x[0])) + ', max switching rate: ' + str(
                        np.exp(x[1])) + ', beta: ' + str(
                        np.exp(x[2])) + ', musens: ' + str(x[3]))
                noiserange = (np.exp(x[1]) - np.exp(x[0])) / np.exp(x[0])
                frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq,
                                                  total_time, mean_mu_max,
                                                  switch_rate=np.exp(x[0]), mu_dep=True, noiserange=noiserange,
                                                  beta=np.exp(x[2]), musens=x[3], mumax=mumax, dependency='sigmoid')
                return 1 - frac_max_mu
        else:
            def fun_sigmoid_switching(x):
                """x contains as entries: minswitchrate, maxswitchrate, beta, musens"""
                if verbose_optimization:
                    print('min switching rate: ' + str(switchratemin) + ', max switching rate: ' + str(
                        switchratemax) + ', beta: ' + str(np.exp(x[0])) + ', musens: ' + str(x[1]))
                noiserange = (switchratemax - switchratemin) / switchratemin
                frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq,
                                                  total_time, mean_mu_max,
                                                  switch_rate=switchratemin, mu_dep=True, noiserange=noiserange,
                                                  beta=np.exp(x[0]), musens=x[1], mumax=mumax, dependency='sigmoid')
                return 1 - frac_max_mu
        """--------------------------------------------------------------------------------------------------------"""

        """Perform optimization"""""
        # For method we can try both L-BFGS-B and TNC, they produce the same result up until now and should both be
        # suitable when the number of variables becomes large

        """Optimization for constant case"""
        res_const = optimize.minimize(fun_const_switching, np.log(switchrate0),
                                      bounds=[(np.log(switchratemin), np.log(switchratemax))],
                                      method='L-BFGS-B')
        res_const.x = np.exp(res_const.x)

        """Optimization for exponential case"""
        opt_start_exp = [np.log(res_const.x[0]), np.log(res_const.x[0]), np.log(beta0)]
        opt_bounds_exp = [tuple(np.log([switchratemin, switchratemax])),
                          tuple(np.log([maxswitchratemin, maxswitchratemax])),
                          tuple(np.log([betamin, betamax]))]
        res_exp = optimize.minimize(fun_exp_switching, opt_start_exp,
                                    bounds=opt_bounds_exp, method='TNC')
        res_exp.x = np.exp(res_exp.x)

        """Optimization for linear case"""
        opt_start_lin = [np.log(res_const.x[0]), np.log(res_const.x[0])]
        opt_bounds_lin = [tuple(np.log([switchratemin, switchratemax])),
                          tuple(np.log([maxswitchratemin, maxswitchratemax]))]
        res_lin = optimize.minimize(fun_lin_switching, opt_start_lin,
                                    bounds=opt_bounds_lin, method='TNC')
        res_lin.x = np.exp(res_lin.x)

        """Optimization for sigmoid case"""
        if OPT_SIGMOID_SWITCHMAXMIN:
            opt_start = [np.log(res_const.x[0]), np.log(res_const.x[0]), np.log(beta0), musens0]
            opt_bounds = [tuple(np.log([switchratemin, switchratemax])),
                          tuple(np.log([maxswitchratemin, maxswitchratemax])),
                          tuple(np.log([betamin, betamax])), (musensmin, musensmax)]
            res_sigm = optimize.minimize(fun_sigmoid_switching, opt_start,
                                         bounds=opt_bounds, method='TNC')
        else:
            opt_start = [np.log(beta0), musens0]
            opt_bounds = [tuple(np.log([betamin, betamax])), (musensmin, musensmax)]
            res_sigm = optimize.minimize(fun_sigmoid_switching, opt_start,
                                         bounds=opt_bounds, method='TNC')

        # TODO: Remove debug statement
        if abs(res_const.fun - res_sigm.fun) < 1e-6:
            print(*[num_env, num_phen_factor, mean_duration, mumax, spread_dom_mu, deltamax], sep=',')
            # res_sigm.x = [np.log(switchratemin), np.log(maxswitchratemax), np.log(betamax),
            #               np.min(good_mus) * (1 - delta / 2)]
            if DEBUG_SINGLES:
                if True:
                    res_sigm.x = [np.log(switchratemin), np.log(maxswitchratemax), np.log(betamax),
                                  np.min(good_mus) - 2*deltamax / (num_phen)]
                    PLOT_SINGLE_SIMU = True

        res_sigm.x[:-1] = np.exp(res_sigm.x[:-1])

        if OPT_SIGMOID_SWITCHMAXMIN:
            sigm_min = res_sigm.x[0]
            sigm_max = res_sigm.x[1]
            sigm_beta = res_sigm.x[2]
            sigm_musens = res_sigm.x[3]
        else:
            sigm_min = switchratemin
            sigm_max = switchratemax
            sigm_beta = res_sigm.x[0]
            sigm_musens = res_sigm.x[1]

        if PLOT_SINGLE_SIMU:
            plot_single_simulation(fit_mat, env_seq, res_const.x, res_lin.x, res_exp.x, res_sigm.x, mumax, mumin, x0,
                                   num_env,
                                   num_phen, total_time, mean_mu_max)
            PLOT_SINGLE_SIMU = False

        if verbose_results:
            print(
                '\nConstant switching: success = ' + str(
                    res_const.success) + '\nMinimal deviation from max mean growth rate: '
                + str(res_const.fun) + '\nFor switching rate: ' + str(res_const.x[0]))

            print('\nGrowth rate dependent switching (exponential): success = ' + str(res_exp.success) +
                  '\nMinimal deviation from max mean growth rate: ' + str(res_exp.fun) +
                  '\nFor min switching rate = ' + str(res_exp.x[0]) + '\n For max switching rate = ' + str(
                res_exp.x[1]) + '\n For beta = ' + str(res_exp.x[2]))

            print('\nGrowth rate dependent switching (linear): success = ' + str(res_lin.success) +
                  '\nMinimal deviation from max mean growth rate: ' + str(res_lin.fun) +
                  '\nFor min switching rate = ' + str(res_lin.x[0]) + '\n For max switching rate = ' + str(
                res_lin.x[1]))

            print('\nGrowth rate dependent switching (sigmoid): success = ' + str(res_sigm.success) +
                  '\nMinimal deviation from max mean growth rate: ' + str(res_sigm.fun) +
                  '\nFor min switching rate = ' + str(sigm_min) + '\n For max switching rate = ' + str(
                sigm_max) + '\n For beta = ' + str(sigm_beta) + '\n For musens = ' + str(sigm_musens))

        results_df = results_df.append(
            {'simulation': simulation, 'num_env': num_env, 'num_phen': num_phen, 'mean_duration': mean_duration,
             'total_time': total_time, 'mumax': mumax, 'mean_mu_max': mean_mu_max,
             'spread_dom_mu': spread_dom_mu, 'deltamax': deltamax, 'switchratemin': switchratemin,
             'switchratemax': switchratemax,
             'G_const': (1 - res_const.fun) * mean_mu_max, 'G_lin': (1 - res_lin.fun) * mean_mu_max,
             'G_exp': (1 - res_exp.fun) * mean_mu_max, 'G_sigm': (1 - res_sigm.fun) * mean_mu_max,
             'Gadv_lin': np.log((1 - res_lin.fun) / (1 - res_const.fun)),
             'Gadv_exp': np.log((1 - res_exp.fun) / (1 - res_const.fun)),
             'Gadv_sigm': np.log((1 - res_sigm.fun) / (1 - res_const.fun)),
             'const': res_const.x[0],
             'exp_min': res_exp.x[0], 'exp_max': res_exp.x[1], 'exp_beta': res_exp.x[2], 'lin_min': res_lin.x[0],
             'lin_max': res_lin.x[1], 'sigm_min': sigm_min, 'sigm_max': sigm_max, 'sigm_beta': sigm_beta,
             'sigm_musens': sigm_musens}, ignore_index=True)

    """Tidy up dataframe"""
    value_vars = ['Gadv_lin', 'Gadv_exp', 'Gadv_sigm']
    id_vars = [colname for colname in results_df.columns if colname not in value_vars]
    results_df_tidy = pd.melt(results_df, id_vars=id_vars, value_vars=value_vars, value_name='Gadv',
                              var_name='mudep_type_Gadv')

    """Store results"""
    results_df_tidy.to_csv(os.path.join(workingdir, "results", "kl_parameterscan_overlap.csv"), index=False,
                           header=True)
else:
    results_df_tidy = pd.read_csv(os.path.join(workingdir, "results", "kl_parameterscan_overlap.csv"))

results_df_full = results_df_tidy.copy()
results_df_tidy['num_phen_factor'] = results_df_tidy['num_phen'] / results_df_tidy['num_env']
results_df_tidy = results_df_tidy[results_df_tidy['num_phen_factor']>0]

custom_colors = ['#d7191c', '#fdae61', '#abd9e9']
blackish = '#%02x%02x%02x' % (35, 31, 32)
scattersize = 20
sns.set_palette(sns.color_palette(custom_colors))
plt.figure(1)
sns.scatterplot(data=results_df_tidy, x='num_env', y='Gadv', hue='mudep_type_Gadv', alpha=0.5, s=scattersize)
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_numenv_overlap.png"))

plt.figure(2)
sns.scatterplot(data=results_df_tidy, x='mean_duration', y='Gadv', hue='mudep_type_Gadv', alpha=0.5, s=scattersize)
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_envduration_overlap.png"))

plt.figure(3)
sns.scatterplot(data=results_df_tidy, x='deltamax', y='Gadv', hue='mudep_type_Gadv', alpha=0.5, s=scattersize)
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_growthrateseparation_overlap.png"))

plt.figure(4)
sns.scatterplot(data=results_df_tidy, x='simulation', y='Gadv', hue='mudep_type_Gadv', alpha=0.5, s=scattersize)
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_simulation_overlap.png"))

plt.figure(5)
sns.scatterplot(data=results_df_tidy, x='G_const', y='Gadv', hue='mudep_type_Gadv', alpha=0.5, s=scattersize)
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_G_const_overlap.png"))

g = sns.FacetGrid(data=results_df_tidy, col="mudep_type_Gadv", hue='num_phen_factor',
                  legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "G_const", "Gadv", s=scattersize)
g.add_legend()
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_panel_overlap.png"))
plt.savefig(os.path.join(workingdir, "results", "kl_parameterscan_panel_overlap.svg"))

g = sns.FacetGrid(data=results_df_tidy, col="mudep_type_Gadv", hue='deltamax', legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "G_const", "Gadv", s=scattersize)
g.add_legend()

g = sns.FacetGrid(data=results_df_tidy, col="mudep_type_Gadv", hue='num_env', legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "G_const", "Gadv", s=scattersize)
g.add_legend()

g = sns.FacetGrid(data=results_df_tidy, col="mudep_type_Gadv", hue='mudep_type_Gadv',
                  legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "Gadv", "num_env", s=scattersize)
g.add_legend()

results_df_new = results_df_tidy.copy()
results_df_new['ratio_spreadmun_mud'] = results_df_new['deltamax'] / results_df_new['spread_dom_mu']
g = sns.FacetGrid(data=results_df_new, col="mudep_type_Gadv", hue='mudep_type_Gadv',
                  legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "Gadv", "ratio_spreadmun_mud", s=scattersize)
g.add_legend()

results_df_new2 = results_df_new.copy()
results_df_new2['switchrate_over_min'] = results_df_new2['const'] * results_df_new2['num_phen'] * 25000
g = sns.FacetGrid(data=results_df_new2, col="mudep_type_Gadv", hue='mudep_type_Gadv',
                  legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "switchrate_over_min", "Gadv", s=scattersize)
g.add_legend()

g = sns.FacetGrid(data=results_df_new2, col="mudep_type_Gadv", hue='mudep_type_Gadv',
                  legend_out=True)  # , height=4, aspect=.5
g.map(sns.scatterplot, "num_phen_factor", "Gadv", s=scattersize)
g.add_legend()

plt.show()
