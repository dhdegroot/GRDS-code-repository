import os
import time

import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt

# import warnings
from kl_simulation_support import *
from scipy.stats import entropy


# warnings.filterwarnings(action="error", category=np.ComplexWarning)

def sigmoid_switch(mu, mumax, H0, beta, musens, noiserange):
    return noiserange * H0 * (mumax - mu) ** beta / ((mumax - musens) ** beta + (mumax - mu) ** beta) + H0


def exponential_mudep(mu, mumin, H0, beta, noiserange):
    return noiserange * H0 * np.exp(-beta * (mu - mumin)) + H0


def linear_mudep(mu, mumin, mumax, H0, noiserange):
    return H0 + ((mumax-mu)/(mumax-mumin)) * noiserange * H0


def switching_mat(mat_fit, num_env, num_phen, switch_rate=0.1, mu_dep=False, noiserange=4, beta=1, musens=0.5, mumax=2,
                  mumin=0, dependency='sigmoid', sensing_cost=0, switch_matrices=None):
    """
    Function which creates a switching matrix with. This matrix is combined with the fitness matrix and
    returned as one matrix.
    :return:
    :param musens:
    :param mumax:
    :param noiserange:
    :param mat_fit: growth rate matrix
    :param num_env: number of phenotypes
    :param num_phen: number of environments
    :param switch_rate: switching rate
    :param mu_dep: boolean to indicate whether the switching rate should depend on mu
    :param beta: constant for tuning sensitivity to mu in case of mu dependency
    :return a: growth/switching matrix
    """
    # First set all away-switching rates, then sum them in diagonal
    cost_mat = np.zeros((num_env, num_phen, num_phen))
    if switch_matrices is not None:
        return mat_fit + switch_matrices - cost_mat
    if mu_dep:
        if dependency == 'sigmoid':
            mat_switch = np.zeros((num_env, num_phen, num_phen))
            if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = sigmoid_switch(mat_fit[i, j, j], mumax, switch_rate, beta, musens,
                                                             noiserange)
                        mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
        elif dependency == 'exponential':
            mat_switch = np.zeros((num_env, num_phen, num_phen))
            if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = exponential_mudep(mat_fit[i, j, j], mumin, switch_rate, beta, noiserange)
                        mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
        elif dependency == 'linear':
            mat_switch = np.zeros((num_env, num_phen, num_phen))
            if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = linear_mudep(mat_fit[i, j, j], mumin, mumax, switch_rate, noiserange)
                        mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
        elif dependency == 'sensing':
            mat_switch = np.zeros((num_env, num_phen, num_phen))
            if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
                for i in range(num_env):
                    np.fill_diagonal(cost_mat[i], sensing_cost)
                    fast_phen = np.argmax(np.diagonal(mat_fit[i]))
                    np.fill_diagonal(mat_switch[i], -switch_rate)
                    mat_switch[i, fast_phen, :] += switch_rate
    else:
        if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
            mat_switch = np.ones((num_env, num_phen, num_phen)) * switch_rate
            for i in range(num_env):
                np.fill_diagonal(mat_switch[i], -(num_phen - 1) * switch_rate)

    return mat_fit + mat_switch - cost_mat


def kl_optimisation(fit_mat, x_0, num_env, num_phen, env_seqs, total_times, mean_mu_max, switch_rate=0.1, mu_dep=False,
                    max_switch_rate=1, beta=1, iterations=10, switch_basis='away'):
    """

    :param: fit_mat:
    :param: b:
    :param: x_0:
    :param: max_mus:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param seed:
    :param max_switch_rate:
    :param beta:
    :param min_sim_time:
    :param avg_env_length:
    :param iterations:
    :return: deviation from maximal mean fitness as a fraction.
    """
    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep,
                          max_switch_rate=max_switch_rate, beta=beta, switch_basis=switch_basis)
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)

    mean_mu = np.zeros(iterations)
    for i in range(iterations):
        start = time.time()

        x_t_fracs = np.zeros((num_phen, len(env_seqs[i][0]) + 1))  # initialise array to store phenotype fractions
        x_t_fracs[:, 0] = x_0
        mu = np.zeros(len(env_seqs[i][0]))  # array to store mean and max growth rate after each environment

        # solve system for specific sequence of environments
        for ind, env in enumerate(env_seqs[i][0]):
            c_scale = np.dot(m_inverse[env], x_t_fracs[:, ind])
            x_t_fracs[:, ind + 1], mu[ind], extinction = grow(eig_vecs[env], eig_vals[env], c_scale,
                                                              env_seqs[i][1][ind])
            if extinction:
                return 1

        mean_mu[i] = sum(mu * env_seqs[i][1]) / total_times[i]

        print('environment sequence ' + str(i) + ' finished in ' + str(time.time() - start) + ' seconds')

    frac_max_mu = mean_mu / mean_mu_max

    return 1 - np.average(frac_max_mu)


def kl_simulation(fit_mat, x_0, num_env, num_phen, env_seq, total_time, mean_mu_max, switch_rate=0.1, mu_dep=False,
                  noiserange=4, beta=1, musens=0.5, mumax=2, dependency='sigmoid', sensing_cost=0, switch_matrices=None):
    """

    :param total_times:
    :param mean_mu_max:
    :param fit_mat:
    :param x_0:
    :param env_seq:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param noiserange:
    :param beta:
    :return: deviation from maximal mean fitness as a fraction.
    """
    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep, sensing_cost=sensing_cost,
                          noiserange=noiserange, beta=beta, musens=musens, mumax=mumax, dependency=dependency, switch_matrices=switch_matrices)
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)
    growthcosts = np.zeros(num_env)
    for env in range(num_env):
        try:
            growthcosts[env] = (np.max(fit_mat[env]) - eig_vals[env].max())  # /fit_mat[env,env,env]
        except:
            max_eig_val = eig_vals[env].max()
            if np.imag(max_eig_val) < 1e-10:
                growthcosts[env] = (np.max(fit_mat[env]) - np.real(max_eig_val))  # /fit_mat[env,env,env]
            else:
                growthcosts[env] = (np.max(fit_mat[env]) - eig_vals[env].max())  # /fit_mat[env,env,env]

    x_t_fracs = np.zeros((num_phen, len(env_seq[0]) + 1))  # initialise array to store phenotype fractions
    x_t_fracs[:, 0] = x_0
    mu = np.zeros(len(env_seq[0]))  # array to store mean and max growth rate after each environment

    # Initialize array for storing results
    results = np.zeros((4, len(env_seq[0])))
    # solve system for specific sequence of environments
    for ind_env, env in enumerate(env_seq[0]):
        c_scale = np.dot(m_inverse[env], x_t_fracs[:, ind_env])
        x_t_fracs[:, ind_env + 1], mu[ind_env], extinction, lag = grow_reportlag(eig_vecs[env],
                                                                                             eig_vals[env],
                                                                                             c_scale,
                                                                                             env_seq[1][ind_env])
        results[0, ind_env] = ind_env
        results[1, ind_env] = lag
        results[2, ind_env] = growthcosts[env]
        results[3, ind_env] = (mu[ind_env] * lag) / np.log(2)
        if extinction:
            continue

    mean_mu = sum(mu * env_seq[1]) / total_time
    frac_max_mu = mean_mu / mean_mu_max
    results_df = pd.DataFrame(data=np.transpose(results), columns=['envnumber', 'lag', 'growthcost', 'lag_gen'])
    results_df['meanmu'] = mean_mu
    results_df['frac_max_mu'] = frac_max_mu
    return frac_max_mu, results_df


def kl_optimisation_new(fit_mat, x_0, num_env, num_phen, env_seq, total_time, mean_mu_max, switch_rate=0.1,
                        mu_dep=False, dependency='sigmoid', mumin=0, sensing_cost=0,
                        noiserange=4, beta=1, musens=0.5, mumax=2):
    """

    :param total_times:
    :param mean_mu_max:
    :param fit_mat:
    :param x_0:
    :param env_seq:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param noiserange:
    :param beta:
    :return: deviation from maximal mean fitness as a fraction.
    """
    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep, dependency=dependency,
                          noiserange=noiserange, beta=beta, musens=musens, mumax=mumax, mumin=mumin,
                          sensing_cost=sensing_cost)
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)

    x_t_fracs = np.zeros((num_phen, len(env_seq[0]) + 1))  # initialise array to store phenotype fractions
    x_t_fracs[:, 0] = x_0
    mu = np.zeros(len(env_seq[0]))  # array to store mean and max growth rate after each environment

    # solve system for specific sequence of environments
    for ind_env, env in enumerate(env_seq[0]):
        c_scale = np.dot(m_inverse[env], x_t_fracs[:, ind_env])
        x_t_fracs[:, ind_env + 1], mu[ind_env], extinction = grow(eig_vecs[env], eig_vals[env], c_scale,
                                                                      env_seq[1][ind_env])
        if extinction:
            return 0

    mean_mu = sum(mu * env_seq[1]) / total_time
    frac_max_mu = mean_mu / mean_mu_max
    return frac_max_mu


def get_mu_pdf(fit_mat, x_0, num_env, num_phen, env_seq, total_time, mean_mu_max, switch_rate=0.1, mu_dep=False,
               noiserange=4, beta=1, musens=0.5, mumax=2, mumin=-0.2, n_bins=200, dependency='sigmoid', sensing_cost=0):
    """

    :param total_times:
    :param mean_mu_max:
    :param fit_mat:
    :param x_0:
    :param env_seq:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param noiserange:
    :param beta:
    :return: deviation from maximal mean fitness as a fraction.
    """

    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep,
                          noiserange=noiserange, beta=beta, musens=musens, mumax=mumax, dependency=dependency,
                          mumin=mumin, sensing_cost=sensing_cost)
    lowerbound = np.min(np.sum(a_mat, axis=1))
    murange = mumax - lowerbound
    bincenters = np.linspace(lowerbound - murange / 100, mumax + murange / 100, n_bins)
    binwidth = bincenters[1] - bincenters[0]
    mu_pdf = np.zeros(n_bins)
    t_cur = 0
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)

    x_t_fracs = np.zeros((num_phen, len(env_seq[0]) + 1))  # initialise array to store phenotype fractions
    x_t_fracs[:, 0] = x_0
    mu = np.zeros(len(env_seq[0]))  # array to store mean and max growth rate after each environment

    # solve system for specific sequence of environments
    for ind_env, env in enumerate(env_seq[0]):
        c_scale = np.dot(m_inverse[env], x_t_fracs[:, ind_env])
        x_t_fracs[:, ind_env + 1], mu[ind_env], extinction, x_pdf, t_env = grow_reportpdf(eig_vecs[env],
                                                                                          eig_vals[env],
                                                                                          c_scale,
                                                                                          env_seq[1][ind_env])

        mu_pdf_env = convert_xpdf_to_mupdf(x_pdf, np.sum(a_mat[env], axis=0), binwidth, bincenters)
        mu_pdf = mu_pdf * (t_cur / (t_cur + t_env)) + mu_pdf_env * (t_env / (t_cur + t_env))
        t_cur = t_cur + t_env
        if extinction:
            return 0

    mean_mu = sum(mu * env_seq[1]) / total_time
    return mean_mu, mu_pdf, bincenters


def get_mu_trace(fit_mat, x_0, num_env, num_phen, env_seq, switch_rate=0.1, mu_dep=False, sensing_cost=0,
                 noiserange=4, beta=1, musens=0.5, mumax=2, mumin=-0.2, n_bins=200, dependency='sigmoid'):
    """

    :param total_times:
    :param mean_mu_max:
    :param fit_mat:
    :param x_0:
    :param env_seq:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param noiserange:
    :param beta:
    :return: deviation from maximal mean fitness as a fraction.
    """
    t_cur = 0

    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep, sensing_cost=sensing_cost,
                          noiserange=noiserange, beta=beta, musens=musens, mumax=mumax, dependency=dependency)
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)

    x_t_fracs = np.zeros((num_phen, len(env_seq[0]) + 1))  # initialise array to store phenotype fractions
    x_t_fracs[:, 0] = x_0

    # Initialize arrays to store trace data
    t_trace = [0]
    mu_trace = [0]
    logOD_trace = [0]
    x_trace = [x_0]
    # solve system for specific sequence of environments
    for ind_env, env in enumerate(env_seq[0]):
        c_scale = np.dot(m_inverse[env], x_t_fracs[:, ind_env])
        x_t_fracs[:, ind_env + 1], t_trace_env, mu_trace_env, x_trace_env = grow_reporttrace(eig_vecs[env],
                                                                                             eig_vals[env],
                                                                                             c_scale,
                                                                                             env_seq[1][ind_env])
        timestep = 0.1
        num_steps = max(int(np.ceil(env_seq[1][ind_env] / timestep)), 10)
        times = np.linspace(0, env_seq[1][ind_env], num_steps)
        logOD_trace_env = grow_reportlogOD(eig_vecs[env], eig_vals[env], c_scale, times)
        logOD_trace_env = np.array(logOD_trace_env[1:])
        t_trace.extend(t_trace_env + t_cur)
        t_cur += env_seq[1][ind_env]
        mu_trace.extend(mu_trace_env)
        x_trace.extend(x_trace_env)
        logOD_trace.extend(logOD_trace_env + logOD_trace[-1])

    t_trace = np.array(t_trace[1:])
    mu_trace = np.array(mu_trace[1:])
    x_trace = np.array(x_trace[1:])
    logOD_trace = np.array(logOD_trace[1:])
    return t_trace, mu_trace, x_trace, logOD_trace


def get_logOD_oneenv(fit_mat, x_0, num_env, num_phen, env=0, switch_rate=0.1, mu_dep=False, dependency='sigmoid',
               noiserange=4, beta=1, musens=0.5, mumax=2, times=np.linspace(0,20), sensing_cost=0):
    """
    :param fit_mat:
    :param x_0:
    :param num_env:
    :param num_phen:
    :param switch_rate:
    :param mu_dep:
    :param noiserange:
    :param beta:
    :return: deviation from maximal mean fitness as a fraction.
    """
    a_mat = switching_mat(fit_mat, num_env, num_phen, switch_rate=switch_rate, mu_dep=mu_dep, dependency=dependency,
                          noiserange=noiserange, beta=beta, musens=musens, mumax=mumax, sensing_cost=sensing_cost)
    # TODO: check how close eigenvalues are
    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    m_inverse = np.linalg.inv(eig_vecs)

    # solve system for specific sequence of environments
    c_scale = np.dot(m_inverse[env], x_0)
    logOD_trace = grow_reportlogOD(eig_vecs[env], eig_vals[env], c_scale, times)
    max_eig = eig_vals[env].max()

    return logOD_trace, max_eig


def get_pretty_pdf(bincenters, mu_pdf, smoothing=0.1, n_bins=200):
    mus = bincenters
    freqs = mu_pdf

    bandwidth = smoothing * mus.std() * mus.size ** (-1 / 5.)
    support = np.linspace(min(mus) - bandwidth, max(mus) + bandwidth, n_bins)
    supp_width = support[1] - support[0]

    kernels = []
    for ind, mu_i in enumerate(mus):
        if ind % 1000 == 0:
            print("Calculating mu dist: '{0}' out of '{1}'".format(ind, len(mus)))
        kernel = scipy.stats.norm(mu_i, bandwidth).pdf(support)
        kernel = kernel * freqs[ind]
        kernels.append(kernel)

    density = np.sum(kernels, axis=0)
    density /= scipy.integrate.trapz(density, support)

    return support, density


def get_landscape_df(fit_mat, env_seq):
    num_env = fit_mat.shape[0]
    total_time = np.sum(env_seq[1])
    landscape_df = pd.DataFrame(columns=['environment', 'growthrate', 'occurrence'])
    for env_ind in range(num_env):
        growthrates = np.diag(fit_mat[env_ind])
        env_df = pd.DataFrame(data=growthrates, columns=['growthrate'])
        env_df['environment'] = env_ind
        times_env = [env_seq[1][ind] for ind, env in enumerate(env_seq[0]) if env == env_ind]
        occurrence = np.sum(times_env)/total_time
        env_df['occurrence'] = occurrence

        landscape_df = landscape_df.append(env_df, sort=True)

    return landscape_df


def convert_xpdf_to_mupdf(x_pdf, growthrates, binwidth, bincenters):
    mu_pdf = np.zeros(len(bincenters))
    x_pdf = np.real(x_pdf)
    for phen, freq in enumerate(x_pdf):
        corr_mu = growthrates[phen]
        bin_number = np.where((bincenters - binwidth / 2 <= corr_mu) & (bincenters + binwidth / 2 >= corr_mu))[0]
        mu_pdf[bin_number] += freq
    return mu_pdf


def get_switch_matrix_from_vect(switch_vect, num_phen):
    if int(np.sqrt(len(switch_vect))) != num_phen - 1:
        raise ValueError('Not the right number of switching rates was given.')

    counter = 0
    switch_matrix = np.zeros((num_phen, num_phen))
    for i in range(num_phen):
        for j in range(num_phen):
            if i != j:
                switch_matrix[i, j] = switch_vect[counter]
                counter += 1

    return switch_matrix


def mullerplot_fig2(x_variable, y_variable, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    phen_names = kwargs.pop("phen_names")
    env_seq = kwargs.pop("env_seq")
    env_changes = np.cumsum(env_seq[1])
    if abs(env_changes[-1] - np.max(data['time'])) < 1e-6:
        env_changes = env_changes[:-1]
    data = data.sort_values('time')
    x = np.unique(np.sort(data['time'])).tolist()
    y = []
    for phen in phen_names:
        phen_data = data[data['phenotype'] == phen][['fraction']].values.flatten()
        y.append(phen_data)

    new_colors = kwargs.pop("new_colors")
    ax.stackplot(x, y, colors=new_colors, lw=0.1)
    for change in env_changes:
        plt.plot([change, change], [0, 1.1], color='0.9', lw=2, ls='--')


def plot_landscape_and_noiserelations(fit_mat, env_seq, res_const_x, res_lin_x, res_exp_x, res_sigm_x, mumax, mumin,
                                      store_figs_filename=False, kinds_to_show=['const', 'lin', 'exp', 'sigm']):
    kind_colors = ['#abd9e9', '#d7191c', '#2c7bb6','#fdae61',]
    num_phen = fit_mat[0].shape[0]
    num_env = fit_mat.shape[0]
    if num_phen == 2:
        phen_colors = sns.xkcd_palette(["greyish", "faded green"])
    else:
        phen_colors = sns.color_palette("cubehelix", num_phen)
        phen_colors.reverse()

    landscape_df = get_landscape_df(fit_mat, env_seq)

    # calculate and show the optimal switching rates per growth rate
    growth_rates = np.linspace(np.min(fit_mat), np.max(fit_mat), 100)
    res_array = np.zeros((len(growth_rates), 5))
    res_array[:, 0] = growth_rates
    res_array[:, 1] = res_const_x

    for i, mu in enumerate(growth_rates):
        res_array[i, 2] = sigmoid_switch(mu, mumax, res_sigm_x[0], res_sigm_x[2], res_sigm_x[3],
                                         (res_sigm_x[1] - res_sigm_x[0]) / res_sigm_x[0])
    for i, mu in enumerate(growth_rates):
        res_array[i, 3] = exponential_mudep(mu, mumin, res_exp_x[0], res_exp_x[2],
                                            (res_exp_x[1] - res_exp_x[0]) / res_exp_x[0])

    for i, mu in enumerate(growth_rates):
        res_array[i, 4] = linear_mudep(mu, mumin, mumax, res_lin_x[0],
                                       (res_lin_x[1] - res_lin_x[0]) / res_lin_x[0])

    res_df = pd.DataFrame(data=res_array, columns=['mu', 'const', 'sigm', 'exp', 'lin'])
    #    traces_df_filtered = traces_df[traces_df['switching type'].isin(kinds_to_show)]

    fig, ax = plt.subplots()
    ax.set_xlabel('growth rate')
    ax.set_ylabel('')
    ax.scatter('growthrate', 'environment', s=20, c='gray', data=landscape_df, zorder=1)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    if num_phen == num_env:
        ax.scatter(np.max(np.max(fit_mat, axis=1), axis=1), range(fit_mat.shape[0]), s=30, c=phen_colors, zorder=2)

    ax2 = ax.twinx()
    ax2.set_ylabel('switch rate')
    for kind_ind, kind in enumerate(kinds_to_show):
        ax2.plot('mu', kind, color=kind_colors[kind_ind], lw=2, data=res_df, zorder=-1)

    ax2.tick_params(axis='y', which='major', left=True, right=False, labelleft=True, labelright=False)
    ax2.tick_params(axis='y', which='minor', left=False, right=False, labelleft=True, labelright=False)
    ax2.yaxis.set_label_position("left")
    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_fitnesslandscape'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))


def plot_mu_distributions(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max, mumax, mumin,
                          store_figs_filename=False,
                          res_const_x=[], res_lin_x=[], res_exp_x=[], res_sigm_x=[], res_sensing_x=[],
                          kinds_to_show=['const', 'lin', 'exp', 'sigm'], sensing_cost=0):
    kind_colors = {'sensing': '#d7191c', 'lin': '#fdae61', 'const': '#abd9e9', 'exp': '#fdae61', 'sigm': '#abd9e9'}
    nbins = 200
    info_dict = {}
    # General
    res_dict = {'const': res_const_x, 'lin': res_lin_x, 'exp': res_exp_x, 'sigm': res_sigm_x, 'sensing': res_sensing_x}
    dep_dict = {'const': '', 'lin': 'linear', 'exp': 'exponential', 'sigm': 'sigmoid', 'sensing': 'sensing'}

    pdfs_df = pd.DataFrame(columns=['growth rate', 'probability', 'switching type'])
    for sw_type in kinds_to_show:
        mu_dep = (sw_type != 'const')
        res = res_dict[sw_type]
        dep = dep_dict[sw_type]
        switch_rate = res[0]
        noiserange, beta, musens = (np.nan, np.nan, np.nan)
        if len(res) > 1:
            noiserange = (res[1] - res[0]) / res[0]
        if len(res) > 2:
            beta = res[2]
        if len(res) > 3:
            musens = res[3]

        mean_mu, mu_pdf, bincenters = get_mu_pdf(fit_mat, x0, num_env, num_phen, env_seq, total_time,
                                                 mean_mu_max, sensing_cost=sensing_cost,
                                                 switch_rate=switch_rate, mu_dep=mu_dep,
                                                 noiserange=noiserange, mumin=mumin,
                                                 beta=beta, musens=musens, mumax=mumax,
                                                 n_bins=nbins, dependency=dep)
        bincenters_pretty, mu_pdf_pretty = get_pretty_pdf(bincenters, mu_pdf, smoothing=0.1, n_bins=nbins)
        pdf_df = pd.DataFrame(
            data=np.concatenate((bincenters_pretty[:, np.newaxis], mu_pdf_pretty[:, np.newaxis]), axis=1),
            columns=['growth rate', 'probability'])
        pdf_df['switching type'] = sw_type
        info_dict.update({sw_type: {'mean_mu': mean_mu}})
        pdfs_df = pdfs_df.append(pdf_df, ignore_index=True)

    pdfs_df_filtered = pdfs_df[pdfs_df['switching type'].isin(kinds_to_show)]

    # Plot the bunch
    fig, ax = plt.subplots()
    ax.set_xlabel('growth rate')
    ax.set_ylabel('probability density')
    for switch_ind, switchtype in enumerate(pdfs_df_filtered['switching type'].unique().tolist()):
        ax.plot('growth rate', 'probability', color=kind_colors[switchtype], lw=2,
                data=pdfs_df_filtered[pdfs_df_filtered['switching type'] == switchtype], label=switchtype)

    ylims = ax.get_ylim()
    for switch_ind, switchtype in enumerate(kinds_to_show):
        ax.plot([info_dict[switchtype]['mean_mu'], info_dict[switchtype]['mean_mu']], [ylims[0] - 10, ylims[1] + 10],
                color=kind_colors[switchtype],
                lw=2, ls='--')
    ax.set_ylim(ylims)

    plt.legend()
    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_mupdf'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))


def plot_mu_trace(fit_mat, x0, num_env, num_phen, env_seq, mumax, mumin, res_const_x=[], res_lin_x=[], res_exp_x=[],
                  res_sigm_x=[], res_sensing_x=[], envs_to_show=10, kinds_to_show=['const', 'lin', 'exp', 'sigm'],
                  store_figs_filename=False, sensing_cost=0, phen_colors=None):
    env_seq = (env_seq[0][:envs_to_show], env_seq[1][:envs_to_show])
    kind_colors = {'sensing': '#d7191c', 'lin': '#fdae61', 'const': '#abd9e9', 'exp': '#fdae61', 'sigm': '#abd9e9'}

    blackish = '#%02x%02x%02x' % (35, 31, 32)

    """Plot traces of average mu over environment"""
    traces_df = pd.DataFrame(columns=['time', 'mu', 'logOD', 'entropy', 'switching type'])
    colnames = ['time']
    for i in range(num_phen):
        colnames.append('phenotype ' + str(i))

    frac_traces_df = pd.DataFrame(columns=['time', 'switching type', 'phenotype', 'fraction'])

    res_dict = {'const': res_const_x, 'lin': res_lin_x, 'exp': res_exp_x, 'sigm': res_sigm_x, 'sensing': res_sensing_x}
    dep_dict = {'const': '', 'lin': 'linear', 'exp': 'exponential', 'sigm': 'sigmoid', 'sensing': 'sensing'}
    for sw_type in kinds_to_show:
        mu_dep = (sw_type != 'const')
        res = res_dict[sw_type]
        dep = dep_dict[sw_type]
        switch_rate = res[0]
        noiserange, beta, musens = (np.nan, np.nan, np.nan)
        if len(res) > 1:
            noiserange = (res[1] - res[0]) / res[0]
        if len(res) > 2:
            beta = res[2]
        if len(res) > 3:
            musens = res[3]

        t_trace, mu_trace, frac_trace, logODtrace = get_mu_trace(fit_mat, x0, num_env, num_phen, env_seq,
                                                                               switch_rate=switch_rate, mu_dep=mu_dep,
                                                                               noiserange=noiserange, beta=beta,
                                                                               sensing_cost=sensing_cost, musens=musens,
                                                                               mumax=mumax,
                                                                               mumin=mumin, dependency=dep)

        entropytrace = np.array([entropy(frac) for frac in frac_trace])

        trace_df = pd.DataFrame(data=np.concatenate((t_trace[:, np.newaxis], mu_trace[:, np.newaxis],
                                                     logODtrace[:, np.newaxis], entropytrace[:, np.newaxis]), axis=1),
                                columns=['time', 'mu', 'logOD', 'entropy'])
        trace_df['switching type'] = sw_type

        frac_trace_df = pd.DataFrame(data=np.concatenate((t_trace[:, np.newaxis], frac_trace), axis=1),
                                     columns=colnames)
        frac_trace_df['switching type'] = sw_type
        frac_trace_df_tidy = pd.melt(frac_trace_df, id_vars=['time', 'switching type'], var_name='phenotype',
                                     value_name='fraction')

        traces_df = traces_df.append(trace_df, ignore_index=True)
        frac_traces_df = frac_traces_df.append(frac_trace_df_tidy, ignore_index=True)

    traces_df_filtered = traces_df[traces_df['switching type'].isin(kinds_to_show)]
    frac_traces_df_filtered = frac_traces_df[frac_traces_df['switching type'].isin(kinds_to_show)]

    fig, ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('growth rate')
    for switch_ind, switchtype in enumerate(kinds_to_show):
        ax.plot('time', 'mu', color=kind_colors[switchtype], lw=2,
                data=traces_df_filtered[traces_df_filtered['switching type'] == switchtype], label=switchtype)

    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_timetraces'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))

    fig, ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('log(population size)')
    for switch_ind, switchtype in enumerate(kinds_to_show):
        ax.plot('time', 'logOD', color=kind_colors[switchtype], lw=2,
                data=traces_df_filtered[traces_df_filtered['switching type'] == switchtype], label=switchtype)

    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_logODtraces'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlabel('time')
    ax.set_ylabel('entropy of phenotypic heterogeneity')
    for switch_ind, switchtype in enumerate(kinds_to_show):
        ax.plot('time', 'entropy', color=kind_colors[switchtype], lw=2,
                data=traces_df_filtered[traces_df_filtered['switching type'] == switchtype], label=switchtype)

    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_entropytraces'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))

    "Make Muller-type plot"
    sns.set_style("white")
    if phen_colors == None:
        if num_phen == 2:
            phen_colors = sns.xkcd_palette(["greyish", "faded green"])
        else:
            phen_colors = sns.color_palette("cubehelix", num_phen)
            phen_colors.reverse()
    g = sns.FacetGrid(frac_traces_df_filtered, col='switching type', legend_out=True,aspect=4)
    g = g.map_dataframe(mullerplot_fig2, 'dummy1', 'dummy2', new_colors=phen_colors, phen_names=colnames[1:],
                        env_seq=env_seq)

    g.set_titles('{col_name}')
    g.set_axis_labels('time', '')
    sns.despine(left=True)

    if store_figs_filename:
        workingdir = os.getcwd()
        filename = store_figs_filename + '_mullerplots'
        plt.savefig(os.path.join(workingdir, "results", filename + '.png'))
        plt.savefig(os.path.join(workingdir, "results", filename + '.svg'))


def plot_single_simulation(fit_mat, env_seq, res_const_x, res_lin_x, res_exp_x, res_sigm_x, mumax, mumin, x0, num_env, num_phen,
                           total_time, mean_mu_max, store_figs_filename=False, envs_to_show=3,
                           kinds_to_show=['const', 'lin', 'exp', 'sigm']):
    # Comment following line if you do not wat to store figures
    plot_landscape_and_noiserelations(fit_mat, env_seq, res_const_x, res_lin_x, res_exp_x, res_sigm_x, mumax, mumin,
                                      store_figs_filename=store_figs_filename, kinds_to_show=kinds_to_show)

    plot_mu_distributions(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max, mumax, mumin, res_const_x=res_const_x, res_lin_x=res_lin_x,
                          res_exp_x=res_exp_x,
                          res_sigm_x=res_sigm_x, store_figs_filename=store_figs_filename,
                          kinds_to_show=kinds_to_show)

    plot_mu_trace(fit_mat, x0, num_env, num_phen, env_seq, mumax, mumin, res_const_x=res_const_x, res_lin_x=res_lin_x,
                  res_exp_x=res_exp_x,
                  res_sigm_x=res_sigm_x, envs_to_show=envs_to_show, kinds_to_show=kinds_to_show,
                  store_figs_filename=store_figs_filename)

    plt.show()
    plt.close('all')


def plot_single_simulation_nolandscape(fit_mat, env_seq, mumax, mumin, x0, num_env, num_phen,
                                       total_time, mean_mu_max, res_const_x=[], res_lin_x=[], res_exp_x=[],
                                       res_sigm_x=[], res_sensing_x=[], store_figs_filename=False, envs_to_show=3,
                                       kinds_to_show=['const', 'lin', 'exp', 'sigm'], sensing_cost=0, phen_colors=None):
    plot_mu_distributions(fit_mat, x0, num_env, num_phen, env_seq, total_time, mean_mu_max, mumax, mumin,
                          res_const_x=res_const_x, res_lin_x=res_lin_x, res_exp_x=res_exp_x, sensing_cost=sensing_cost,
                          res_sigm_x=res_sigm_x, res_sensing_x=res_sensing_x, store_figs_filename=store_figs_filename,
                          kinds_to_show=kinds_to_show)

    plot_mu_trace(fit_mat, x0, num_env, num_phen, env_seq, mumax, mumin, res_const_x=res_const_x, res_lin_x=res_lin_x,
                  res_exp_x=res_exp_x,
                  res_sigm_x=res_sigm_x, res_sensing_x=res_sensing_x, envs_to_show=envs_to_show,
                  kinds_to_show=kinds_to_show, phen_colors=phen_colors,
                  store_figs_filename=store_figs_filename, sensing_cost=sensing_cost)

    plt.show()
    plt.close('all')
