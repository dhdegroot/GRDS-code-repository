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
RANDOM_TIMES = False
PLOT_DETAILS = True
SIM_TIME_FACTOR = 1
workingdir = os.getcwd()
# Set parameter values
num_env = 20
num_phen = num_env
num_iter = 1
num_switch = 100
# num_betas = 3
mean_duration = 10
mu_max = 1
# sp_sensing_cost = 0.01
maxadaptationrate = 10000
upper_limit_bad = 0.0625  # Bad growth rates are drawn from [0, upper_limit_bad)
growth_dep_factor = [10, 100]

kinds = ['const', 'lin10', 'lin100']

kinds_dict = {'const': 'constant', 'lin10': 'linear', 'lin100': 'linear'}
labels = {'const': 'bet-hedging', 'lin10': 'GRDS (strength 10)', 'lin100': 'GRDS (strength 100)'}
switchmax = maxadaptationrate / (num_phen - 1)
switchmin = switchmax / 10000000

# sensing_cost = sp_sensing_cost * num_env
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

"""Create rest of system"""
# Make markov transition matrix
# Either random
# trans_mat = np.random.rand(num_env, num_env)
# Or just all equal
trans_mat = np.ones((num_env, num_env))
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

# Gather some useful information
mumaxs = np.asarray([np.max(fitness_mat) for fitness_mat in fit_mat])
mumax = np.max(mumaxs)  # Fastest growth possible
mean_mu_max = sum(mumaxs[env_seq[0]] * env_seq[1]) / total_time  # Average growth Darwinian devil
mumin = np.min(fit_mat)  # Slowest growth rate possible (or zero)

# Choose switch rates and betas which should be tried
switchrates = np.exp(np.linspace(np.log(switchmin), np.log(switchmax), num_switch))

if num_iter == 1:
    landscape_df = get_landscape_df(fit_mat, env_seq)
"""--------------------------------------------------------------------------------------------------------"""
iter_id = 0
results_df = pd.DataFrame(
    columns=['envnumber', 'lag', 'growthcost', 'lag_gen', 'meanmu', 'frac_max_mu', 'switchmin', 'switchmax', 'beta',
             'musens', 'dependency', 'kind', 'meandelay', 'meandelay_gen', 'meangrowthcost', 'simulation'])
mean_df = pd.DataFrame(
    columns=['kind', 'dependency', 'meandelay', 'meandelay_gen', 'meangrowthcost', 'meanmu', 'frac_max_mu',
             'num_env', 'num_phen', 'simulation'])

x0 = np.zeros(num_phen)
x0[-1] = 1

"""Perform simulation for constant switching rates"""
for ind_switch in range(num_switch):
    if (ind_switch + 1) % 10 == 0:
        print("Currently performing iteration " + str(ind_switch + 1) + " out of " + str(num_switch))
    switch_rate = switchrates[ind_switch]
    res_const_x = [switch_rate]
    res_lin10_x = [switch_rate / growth_dep_factor[0], switch_rate]
    res_lin100_x = [switch_rate / growth_dep_factor[1], switch_rate]
    res_sensing_x = [maxadaptationrate]
    res_dict = {'const': res_const_x, 'lin10': res_lin10_x, 'lin100': res_lin100_x}
    for kind in kinds:
        dep = kinds_dict[kind]
        mu_dep = (kind not in ['const'])
        res = res_dict[kind]
        switch_rate = res[0]
        noiserange, beta, musens, switchmax = (np.nan, np.nan, np.nan, np.nan)
        if len(res) > 1:
            noiserange = (res[1] - res[0]) / res[0]
            switchmax = res[1]
        if len(res) > 2:
            beta = res[2]
        if len(res) > 3:
            musens = res[3]

        dummy, results_df_kind = kl_simulation(fit_mat, x0, num_env, num_phen, env_seq,
                                               total_time, mean_mu_max, switch_rate, mu_dep=mu_dep,
                                               dependency=dep, noiserange=noiserange, beta=beta, musens=musens,
                                               mumax=mumax)

        results_df_kind['switchmin'] = switch_rate
        results_df_kind['switchmax'] = switchmax
        results_df_kind['beta'] = beta
        results_df_kind['musens'] = musens
        results_df_kind['dependency'] = dep
        results_df_kind['kind'] = kind
        meandelay = np.mean(results_df_kind['lag'])
        results_df_kind['meandelay'] = meandelay
        meandelay_gen = np.mean(results_df_kind['lag_gen'])
        results_df_kind['meandelay_gen'] = meandelay_gen
        meangrowthcost = np.mean(results_df_kind['growthcost'])
        results_df_kind['meangrowthcost'] = meangrowthcost
        results_df_kind['simulation'] = ind_switch

        mean_df = mean_df.append(
            {'kind': kind, 'dependency': dep, 'meandelay': meandelay, 'meandelay_gen': meandelay_gen,
             'meangrowthcost': meangrowthcost, 'meanmu': results_df_kind['meanmu'][0],
             'frac_max_mu': results_df_kind['frac_max_mu'][0],
             'num_env': num_env, 'num_phen': num_phen, 'simulation': ind_switch},
            ignore_index=True)
        results_df = results_df.append(results_df_kind)

"""---------------------------------------------------------------------------------------------------------"""

"""Perform simulations for mu-dependent switching rates"""
# print('Started simulation for mu-dependent switching rates')
# for ind_iter in range(num_iter):
#     for ind_switch in range(num_switch):
#         noiserange = (switchmax - switchrates[ind_switch]) / switchrates[ind_switch]
#         for ind_beta in range(num_betas):
#             start = time.time()
#             dummy, results_df_simu = kl_simulation(fit_mats[ind_iter], x0, num_env, num_phen, env_seqs[ind_iter],
#                                                    total_times[ind_iter], mean_mu_max[ind_iter],
#                                                    switch_rate=switchrates[ind_switch], mu_dep=True,
#                                                    noiserange=noiserange,
#                                                    beta=betas[ind_beta], musens=musens, mumax=mumax)
#
#             results_df_simu['switchrate'] = switchrates[ind_switch]
#             results_df_simu['beta'] = betas[ind_beta]
#             results_df_simu['iteration'] = ind_iter
#             meanlag = np.mean(results_df_simu['lag'])
#             results_df_simu['meanlag'] = meanlag
#             meangrowthcost = np.mean(results_df_simu['growthcost'])
#             results_df_simu['meangrowthcost'] = meangrowthcost
#             results_df_simu['iter_id'] = iter_id
#             mean_df = mean_df.append(
#                 {'iter_id': iter_id, 'iteration': ind_iter, 'switchrate': switchrates[ind_switch],
#                  'betalabel': 'beta=' + str(betas[ind_beta]), 'beta': betas[ind_beta], 'meanlag': meanlag,
#                  'meangrowthcost': meangrowthcost, 'meanmu': results_df_simu['meanmu'][0],
#                  'frac_max_mu': results_df_simu['frac_max_mu'][0]}, ignore_index=True)
#
#             iter_id += 1
#             if (ind_iter * num_switch * num_betas + ind_switch * num_betas + ind_beta + 1) % 10 == 0:
#                 print('Iteration ' + str(
#                     ind_iter * num_switch * num_betas + ind_switch * num_betas + ind_beta + 1) + ' of ' + str(
#                     num_switch * num_iter * num_betas) + ', for mu-dependent switching rates finished in ' + str(
#                     time.time() - start) + ' seconds')
#             results_df = results_df.append(results_df_simu, sort=True)

"""---------------------------------------------------------------------------------------------------------"""
results_df_tidy = pd.melt(results_df,
                          id_vars=['envnumber', 'frac_max_mu', 'meanmu', 'switchmin', 'switchmax', 'simulation', 'beta',
                                   'musens', 'dependency', 'meandelay', 'meandelay_gen', 'meangrowthcost'])

# """Make scatterplot of x=mean lagtimes, y=mean growth cost (per simulation), color=beta"""
# plt.figure(1)
# ax = sns.scatterplot(x='meanlag', y='meangrowthcost', hue='beta', data=mean_df, legend='brief', size='frac_max_mu',
#                      sizes=(20, 200))
#
# L = ax.legend()
# L.get_texts()[0].set_text("Beta")
# ax.set_title("Growth tradeoff")
# ax.set_xlabel("Mean lag-time over sequence of environments")
# ax.set_ylabel("Growth cost")
#
# plt.savefig(os.path.join(workingdir, "results", "pareto_lag_growthcost.png"))

custom_colors = ['#abd9e9', '#fdae61', '#d7191c']
# custom_colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
blackish = '#%02x%02x%02x' % (35, 31, 32)
# sns.set(context='notebook', style='white', font='calibri', font_scale=1.2)
"""Make lineplot of x=mean lagtimes, y=mean growth cost (per simulation), color=beta, indicate optimum per 
        pareto-front by point"""
fig, ax = plt.subplots(1, figsize=(6, 5))
# pareto = mean_df
for kind_ind, kind in enumerate(kinds):
    pareto = mean_df[mean_df["kind"] == kind]
    ax.plot(pareto['meandelay'], pareto['meangrowthcost'], color=custom_colors[kind_ind], lw=2, zorder=-1,
            label=labels[kind])
    optimum = pareto.iloc[np.argmax(pareto['frac_max_mu'].values), :]
    ax.scatter(x=optimum['meandelay'], y=optimum['meangrowthcost'], c=custom_colors[kind_ind], s=120, zorder=1)
    ax.annotate("   G = {:.2f}".format(optimum['frac_max_mu']), (optimum['meandelay'], optimum['meangrowthcost']))

ax.legend()
ax.set_title("growth trade-off")
ax.set_xlabel("mean growth delay")
ax.set_ylabel("mean growth rate cost")

plt.savefig(os.path.join(workingdir, "results", "pareto_lag_growthcost.png"))
plt.savefig(os.path.join(workingdir, "results", "pareto_lag_growthcost.svg"))

if num_iter == 1:
    """Plot summary of fitness landscape"""
    """Plot relations between growth rate and switching rates"""
    # calculate and show the optimal switching rates per growth rate
    growth_rates = np.linspace(np.min(fit_mat), np.max(fit_mat))
    # res_array = np.zeros((len(growth_rates), 2 + num_betas))
    # res_array[:, 0] = growth_rates
    #
    # # Optimal switch rate for constant case
    # res_array[:, 1] = optima_df[optima_df['beta'] == 0.0]['switchrate'].values[0]
    # colnames = ['growth rate', 'constant']
    # for beta_ind, beta in enumerate(betas):
    #     opt_switchrate = optima_df[optima_df['beta'] == beta]['switchrate'].values[0]
    #     colnames.append('beta=' + str(beta))
    #     noiserange = (switchmax - opt_switchrate) / opt_switchrate
    #     for mu_ind, mu in enumerate(growth_rates):
    #         res_array[mu_ind, 2 + beta_ind] = sigmoid_switch(mu, mumax, opt_switchrate, beta, musens, noiserange)
    # res_df = pd.DataFrame(data=res_array, columns=colnames)
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel('growth rate')
    # ax.set_ylabel('')
    # ax.scatter('growthrate', 'environment', s=5, c='gray', data=landscape_df, zorder=-1)
    # ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    #
    # ax2 = ax.twinx()
    # ax2.set_ylabel('switch rate')
    # ax2.plot('growth rate', 'constant', color=custom_colors[0], lw=4, data=res_df)
    # for beta_ind in range(num_betas):
    #     ax2.plot('growth rate', colnames[2 + beta_ind], color=custom_colors[beta_ind + 1], lw=4, data=res_df)
    # ax2.tick_params(axis='y', which='major', left=True, right=False, labelleft=True, labelright=False)
    # ax2.tick_params(axis='y', which='minor', left=False, right=False, labelleft=True, labelright=False)
    # ax2.yaxis.set_label_position("left")
    # plt.savefig(os.path.join(workingdir, "results", "pareto_environment_and_growthHrelations.png"))
    # plt.savefig(os.path.join(workingdir, "results", "pareto_environment_and_growthHrelations.svg"))

    """Create an example by having one environment and showing the log OD time courses for all betas"""
    # Choose environment
    env = 0
    env_time = 15
    times = np.linspace(0, env_time)
    logOD_traces = np.zeros((len(times), len(kinds)))
    max_eigs = np.zeros(len(kinds))

    # Calculate time trace for optimal constant switch rate
    for kind_ind, kind in enumerate(kinds):
        max_growth = np.max(mean_df[mean_df['kind'] == kind].meanmu)
        optimal_rows = results_df[(results_df['kind'] == kind) & (results_df['meanmu'] == max_growth)]
        dep = kinds_dict[kind]
        mu_dep = (kind not in ['const'])
        switch_rate = optimal_rows['switchmin'].iloc[0]
        noiserange, beta, musens, switchmax = (np.nan, np.nan, np.nan, np.nan)
        if dep is 'linear':
            switchmax = optimal_rows['switchmax'][0]
            noiserange = (switchmax - switch_rate) / switch_rate

        logOD_traces[:, kind_ind], max_eigs[kind_ind] = get_logOD_oneenv(fit_mat, x0, num_env, num_phen,
                                                                         env=env, dependency=dep,
                                                                         switch_rate=switch_rate,
                                                                         mu_dep=mu_dep, noiserange=noiserange,
                                                                         beta=beta, musens=musens, mumax=mumax,
                                                                         times=times)

    fig4, ax4 = plt.subplots()
    # Plot fastest growth possible in this environment
    for line_ind in range(len(kinds)):
        logOD_trace = logOD_traces[:, line_ind]
        ax4.plot(times, logOD_trace, lw=4, c=custom_colors[line_ind], zorder=1)
        if line_ind == 0:
            maxlogOD = logOD_trace[-1]
            maxmu = np.max(fit_mat[env])
            maxeig = max_eigs[line_ind]
            ax4.plot(times, maxmu * (times - times[-1]) + maxlogOD, '--', lw=1, c=blackish, zorder=-2)
            stationary_tangent = maxeig * (times - times[-1]) + maxlogOD
            ax4.plot(times, stationary_tangent, '-.', lw=1, c=blackish, zorder=-1)
            delay_times = times[stationary_tangent <= 0]
            ax4.plot(delay_times, np.zeros(len(delay_times)), lw=2, c=blackish, zorder=0)

    ax4.set_xlabel('time (a.u.)')
    ax4.set_ylabel('log(cell count) (a.u.)')
    plt.savefig(os.path.join(workingdir, "results", "oneenv_paretoplot.png"))
    plt.savefig(os.path.join(workingdir, "results", "oneenv_paretoplot.svg"))

plt.show()
