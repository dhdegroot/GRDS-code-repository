import os

from scipy import optimize

from kl_paretofront_support import *

"""
In this script we will create one fitness landscape and try to optimize the population growth rate. First, we will
optimize a constant switching rate. Then, we will find the optimal sigmoid. Both will be limited by a maximal, and a 
minimal switching rate.   

To simplify matters we will assume that # environments = # phenotypes.
Phenotype i is the dominant one in environment i. We sample:
- 1 dominant growth rate from a uniform distribution on the interval [mumin,mumax]
- n-1 non-dominant growth rates, s.t. mu_dom - mu_non in [deltamin,deltamax], with deltamin large enough
- order of environments (Markov chain, equal weights)
- environment time of specific instance of environment (from exponential distribution)

The following we just take fixed
- Average environment lengths will be taken equal
- Number of phenotypes

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
# Set parameter values for system
num_env = 2
num_phen = 3
num_iter = 1
avg_env_length = [10] * num_env  # must be a list of length num_env
mu_max = 0.8
sp_sensing_cost = 0.03
maxadaptationrate = 1
upper_limit_bad = 0.05  # Bad growth rates are drawn from [0, upper_limit_bad)
growth_dep_factor = 100

np.random.seed(1337)
env_seq = ([0, 1], avg_env_length)
total_time = sum(env_seq[1])
x0 = np.zeros(num_phen)
x0[2] = 1

sensing_cost = sp_sensing_cost * num_env

"""Create fitness landscapes, either sparse or dense"""
# Create fitness matrix for each environment
# One dominant growth rate per environment given by mu_max
# Non-dominant growth rates are zero
fit_mat = np.zeros((num_env, num_phen, num_phen))
for ind_env in range(num_env):
    bad_mus = np.random.rand(num_phen) * upper_limit_bad
    ind_good_phen = ind_env % num_phen
    np.fill_diagonal(fit_mat[ind_env], bad_mus)
    fit_mat[ind_env, ind_good_phen, ind_good_phen] = mu_max

# Gather some useful information
mumaxs = np.asarray([np.max(fitness_mat) for fitness_mat in fit_mat])
mumax = np.max(mumaxs)  # Fastest growth possible
mean_mu_max = sum(mumaxs[env_seq[0]] * env_seq[1]) / total_time  # Average growth Darwinian devil
mumin = np.min(fit_mat)  # Slowest growth rate possible (or zero)

"""Initialize optimization for constant, linear, exponential, sigmoidal relations"""
# Set optimisation parameters

# minimal switchrate
switchratemax = maxadaptationrate / (num_phen - 1)
switchratemin = switchratemax / 1000
switchrate0 = switchratemin

# maxswitchrate
maxswitchratemax = switchratemax
maxswitchratemin = switchratemin
maxswitchrate0 = maxswitchratemax


"""--------------------------------------------------------------------------------------------------------"""

"""Create dataframe that summarizes fitness landscape"""
landscape_df = get_landscape_df(fit_mat, env_seq)

"""Define optimization functions"""


def fun_const_switching(x):
    print('switching rate: ' + str(np.exp(x)))
    frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max,
                                      switch_rate=np.exp(x),
                                      mu_dep=False)
    return 1 - frac_max_mu


def fun_growthdep_switching(x):
    """x contains as entries: minswitchrate, maxswitchrate, beta, musens"""
    max_switch_rate = np.exp(x)
    min_switch_rate = max_switch_rate / growth_dep_factor
    print('min switching rate: ' + str(min_switch_rate) + ', max switching rate: ' + str(max_switch_rate))
    noiserange = (max_switch_rate - min_switch_rate) / min_switch_rate
    frac_max_mu = kl_optimisation_new(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max,
                                      switch_rate=min_switch_rate, mu_dep=True, noiserange=noiserange, mumin=mumin,
                                      mumax=mumax, dependency='linear', sensing_cost=sensing_cost)

    return 1 - frac_max_mu


"""--------------------------------------------------------------------------------------------------------"""

"""Perform optimization"""""
# For method we can try both L-BFGS-B and TNC, they produce the same result up until now and should both be
# suitable when the number of variables becomes large
res_const = optimize.minimize(fun_const_switching, np.log(switchrate0),
                              bounds=[(np.log(switchratemin), np.log(switchratemax))],
                              method='L-BFGS-B')
res_const.x = np.exp(res_const.x)
res_const_x = res_const.x

"""Optimization for high-low case"""
opt_start = np.log(res_const.x[0])
opt_bounds = [tuple(np.log([switchratemin, switchratemax]))]
res_lin = optimize.minimize(fun_growthdep_switching, opt_start, bounds=opt_bounds, method='L-BFGS-B')
res_lin.x = np.exp(res_lin.x)
res_max_switch_rate = res_lin.x[0]
res_min_switch_rate = res_max_switch_rate/growth_dep_factor

res_lin_x = [res_min_switch_rate, res_max_switch_rate]
res_sensing_x = [maxadaptationrate]

print('\nConstant switching: success = ' + str(res_const.success) + '\nMinimal deviation from max mean growth rate: '
      + str(res_const.fun) + '\nFor switching rate: ' + str(res_const.x))

print('\nGrowth rate dependent switching (high-low): success = ' + str(res_lin.success) +
          '\nMinimal deviation from max mean growth rate: ' + str(res_lin.fun) +
          '\nFor min switching rate = ' + str(res_min_switch_rate) +
          '\n For max switching rate = ' + str(res_max_switch_rate))

"""Plot summary of fitness landscape"""
# env_order = np.unique(landscape_df['occurrence'], return_index=True)[1]
# order = landscape_df['environment'].values[env_order]
sns.set(style='whitegrid')
g0 = sns.catplot(x='growthrate', y='environment', kind='strip', hue='environment', data=landscape_df, legend=False,
                 legend_out=False, orient='h', height=5, aspect=1.2)
g0.set(ylabel='Environments in increasing occurrence', xlabel='Growth rates')
plt.savefig(os.path.join(workingdir, "results", "fitness_landscape.png"))

colors = []
rgb_codes = [[182, 169, 219], [219, 176, 169], [169, 219, 195]]
for rgb_code in rgb_codes:
    colors.append('#%02x%02x%02x' % (rgb_code[0], rgb_code[1], rgb_code[2]))


plot_single_simulation_nolandscape(fit_mat, env_seq, mumax, mumin, x0,
                                   num_env, num_phen, total_time, mean_mu_max, res_const_x=res_const_x,
                                   res_lin_x=res_lin_x, res_sigm_x=[], res_exp_x=[],
                                   store_figs_filename='kl_illustration', sensing_cost=sensing_cost,
                                   envs_to_show=5, phen_colors=colors, res_sensing_x=res_sensing_x,
                                   kinds_to_show=['const', 'lin', 'sensing'])

plt.show()
