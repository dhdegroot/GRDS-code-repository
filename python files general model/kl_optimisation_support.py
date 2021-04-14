import time

from kl_simulation_support import *


def switching_mat(mat_fit, num_env, num_phen, switch_rate=0.1, mu_dep=False, max_switch_rate=1, beta=1,
                  switch_basis='away'):
    """
    Function which creates a switching matrix with. This matrix is combined with the fitness matrix and
    returned as one matrix.
    :param mat_fit: growth rate matrix
    :param num_env: number of phenotypes
    :param num_phen: number of environments
    :param switch_rate: switching rate
    :param mu_dep: boolean to indicate whether the switching rate should depend on mu
    :param max_switch_rate: in case the switching rate depends on mu, this is the maximal rate (for smallest mu)
    :param beta: constant for tuning sensitivity to mu in case of mu dependency
    :return a: growth/switching matrix
    """
    # First set all away-switching rates, then sum them in diagonal
    if mu_dep:
        offset = mat_fit.min()
        mat_switch = np.zeros((num_env, num_phen, num_phen))
        if np.isscalar(max_switch_rate) | (len([max_switch_rate]) == 1):
            for i in range(num_env):
                for j in range(num_phen):
                    mat_switch[i, :, j] = max_switch_rate*np.exp(-(mat_fit[i, j, j]-offset)*beta)
                    mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
        else:
            if switch_basis == 'away':
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = max_switch_rate[j]*np.exp(-(mat_fit[i, j, j]-offset)*beta)
                        mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
            elif switch_basis == 'toward':
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = max_switch_rate*np.exp(-(mat_fit[i, j, j]-offset)*beta)
                        mat_switch[i, j, j] = 0
                        mat_switch[i, j, j] = - np.sum(mat_switch[i, :, j])
            elif switch_basis == 'both':
                max_switch_rate = get_switch_matrix_from_vect(max_switch_rate, num_phen)
                for i in range(num_env):
                    for j in range(num_phen):
                        for k in range(num_phen):
                            if j != k:
                                mat_switch[i, j, k] = max_switch_rate[j, k] * np.exp(-(mat_fit[i, k, k] - offset)*beta)
                    for j in range(num_phen):
                        mat_switch[i, j, j] = - np.sum(mat_switch[i, :, j])

    else:
        if np.isscalar(switch_rate) | (len([switch_rate]) == 1):
            mat_switch = np.ones((num_env, num_phen, num_phen))*switch_rate
            for i in range(num_env):
                np.fill_diagonal(mat_switch[i], -(num_phen - 1) * switch_rate)
        else:
            mat_switch = np.zeros((num_env, num_phen, num_phen))
            if switch_basis == 'away':
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = switch_rate[j]
                        mat_switch[i, j, j] = -(num_phen - 1) * mat_switch[i, j, j]
            elif switch_basis == 'toward':
                for i in range(num_env):
                    for j in range(num_phen):
                        mat_switch[i, :, j] = switch_rate
                        mat_switch[i, j, j] = 0
                        mat_switch[i, j, j] = - np.sum(mat_switch[i, :, j])
            elif switch_basis == 'both':
                switch_rate = get_switch_matrix_from_vect(switch_rate, num_phen)
                for i in range(num_env):
                    for j in range(num_phen):
                        for k in range(num_phen):
                            if j != k:
                                mat_switch[i, j, k] = switch_rate[j, k]
                    for j in range(num_phen):
                        mat_switch[i, j, j] = - np.sum(mat_switch[i, :, j])

    # Return growth/switching matrix
    return mat_fit + mat_switch


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

    frac_max_mu = mean_mu/mean_mu_max

    return 1-np.average(frac_max_mu)


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
