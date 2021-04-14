"""Several supporting functions for the KL simulation"""
import numpy as np


def build_model(num_env=10, num_phen=10, mean_fit=0, std_fit=1, range_switch=(0.01, 1), k_mu_dep=0.1, seed=False):
    """
    Function which creates a fitness matrix with random growth rates on the diagonal and a switching matrix with a
    constant away-switching rate for all phenotypes in all environments (for now). These matrices are combined and
    returned as one matrix. An intial distribution x_0 with 1 for every phenotype is created and returned.
    :param num_env: number of phenotypes
    :param num_phen: number of environments
    :param mean_fit: mean growth rate (for sampling)
    :param std_fit: standard deviation of the growth rate (for sampling)
    :param range_switch: range from witch to draw the switching rate
    :param k_mu_dep: constant which determines sensitivity to mu
    :param seed: seed for PRNG
    :return a: growth/switching matrices
    :return x_0: vector x(0) filled with ones
    :return b: matrix of transition probabilities between environments
    """
    if seed:
        np.random.seed(seed)

    # Initialise
    mat_fit = np.zeros((num_env, num_phen, num_phen))
    mat_switch_const = np.ones((num_env, num_phen, num_phen))
    mat_switch_dep = np.ones((num_env, num_phen, num_phen))

    # Fill diagonal of fitness matrix
    for i in range(num_env):
        np.fill_diagonal(mat_fit[i], np.random.normal(loc=mean_fit, scale=std_fit, size=num_phen))

    # First set all away-switching rates, then sum them in diagonal
    offset = mat_fit.min()
    for i in range(num_env):
        for j in range(num_phen):
            switch_rate = np.random.uniform(range_switch[0], range_switch[1])
            mat_switch_const[i, :, j] = switch_rate
            mat_switch_dep[i, :, j] = switch_rate / (mat_fit[i, j, j] - offset + k_mu_dep)
            mat_switch_dep[i, j, j] = -(num_phen - 1) * mat_switch_dep[i, j, j]
            mat_switch_const[i, j, j] = -(num_phen - 1) * mat_switch_const[i, j, j]

    # Create random transition probability matrix, switching to the same environment is not allowed
    b = np.random.uniform(0, 1, (num_env, num_env))
    np.fill_diagonal(b, 0)
    norm_factors = np.sum(b, 0)
    for i in range(num_env):
        b[:, i] = b[:, i] / norm_factors[i]

    # Return growth/switching matrix, starting distribution and transition probability matrix
    return [mat_fit + mat_switch_const, mat_fit + mat_switch_dep], np.ones(num_phen) / num_phen, b


def grow(eig_vecs, eig_vals, c_scale, t, extinction=False):
    """
    calculate x(t)
    :param eig_vecs: eigenvectors
    :param eig_vals: eigenvalues
    :param c_scale: rescaling constants
    :param t: time
    :param extinction: flag to check whether population is extinct
    :return: x_t/total: x(t) in fractions per phenotype
    :return: mu: average growth rate in this environment
    """
    max_eig = eig_vals.max()
    x_t = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * t))
    total = np.sum(x_t)

    if total <= 10 ** -6:
        extinction = True

    mu = np.log(total) / t + max_eig

    return np.real(x_t / total), np.real(mu), extinction


def grow_reportlag(eig_vecs, eig_vals, c_scale, t, extinction=False):
    """
    calculate x(t)
    :param eig_vecs: eigenvectors
    :param eig_vals: eigenvalues
    :param c_scale: rescaling constants
    :param t: time
    :param extinction: flag to check whether population is extinct
    :return: x_t/total: x(t) in fractions per phenotype
    :return: mu: average growth rate in this environment
    """
    max_eig = eig_vals.max()
    x_t = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * t))
    total = np.sum(x_t)

    if total <= 10 ** -6:
        extinction = True

    mu = np.log(total) / t + max_eig
    x_t_norm = x_t/total
    lag = t - (mu * t) / max_eig
    if (np.imag(mu) < 1e-10) and (np.max(np.abs(np.imag(x_t_norm))) < 1e-10) and (np.imag(lag) < 1e-10):
        mu = np.real(mu)
        x_t_norm = np.real(x_t_norm)
        lag = np.real(lag)
    else:
        pass
    return x_t_norm, mu, extinction, lag


def grow_reportpdf(eig_vecs, eig_vals, c_scale, t, extinction=False, timestep=0.1):
    """
    calculate x(t)
    :param eig_vecs: eigenvectors
    :param eig_vals: eigenvalues
    :param c_scale: rescaling constants
    :param t: time
    :param extinction: flag to check whether population is extinct
    :return: x_t/total: x(t) in fractions per phenotype
    :return: mu: average growth rate in this environment
    """
    num_steps = int(np.ceil(t / timestep))
    times = np.linspace(0, t, num_steps)
    x_pdf = np.zeros(len(c_scale))
    t_cur = 0

    max_eig = eig_vals.max()
    for ind_time in range(1,num_steps):
        new_time = times[ind_time]
        new_pdf = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * t_cur))
        new_pdf = new_pdf/np.sum(new_pdf)

        x_pdf = x_pdf*(t_cur/new_time) + new_pdf*((new_time-t_cur)/new_time)
        t_cur = new_time

    x_t = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * t))
    total = np.sum(x_t)

    if total <= 10 ** -6:
        extinction = True

    mu = np.log(total) / t + max_eig
    x_t_norm = x_t/total
    lag = t - (mu * t) / max_eig
    if (np.imag(mu) < 1e-10) and (np.max(np.abs(np.imag(x_t_norm))) < 1e-10) and (np.imag(lag) < 1e-10) and (np.max(np.abs(np.imag(x_pdf))) < 1e-10):
        mu = np.real(mu)
        x_t_norm = np.real(x_t_norm)
        lag = np.real(lag)
        x_pdf = np.real(x_pdf)
    else:
        pass
    return x_t_norm, mu, extinction, x_pdf, t_cur


def grow_reporttrace(eig_vecs, eig_vals, c_scale, t, extinction=False, timestep=0.1):
    """
    calculate x(t)
    :param eig_vecs: eigenvectors
    :param eig_vals: eigenvalues
    :param c_scale: rescaling constants
    :param t: time
    :param extinction: flag to check whether population is extinct
    :return: x_t/total: x(t) in fractions per phenotype
    :return: mu: average growth rate in this environment
    """
    num_steps = max(int(np.ceil(t / timestep)), 10)
    times = np.linspace(0, t, num_steps)
    t_diff = times[1]-times[0]

    t_trace = times
    mu_trace = np.zeros(len(times))
    x_trace = np.zeros((num_steps, len(c_scale)))
    x_trace[0, :] = np.sum(np.multiply(c_scale, eig_vecs), axis=1)

    max_eig = eig_vals.max()
    for ind_time in range(1, num_steps):
        new_time = times[ind_time]
        x_trace[ind_time, :] = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * new_time))

    x_t = np.dot(np.multiply(c_scale, eig_vecs), np.exp((eig_vals - max_eig) * t))
    total = np.sum(x_t)

    if total <= 10 ** -6:
        extinction = True

    frac_trace = x_trace/np.tile(np.sum(x_trace, axis=1)[:, np.newaxis], (1, len(c_scale)))
    mu_trace[1:] = (np.diff(np.log(np.sum(x_trace, axis=1)))/t_diff)+max_eig

    t_trace = t_trace[1:]
    mu_trace = mu_trace[1:]
    frac_trace = frac_trace[1:]

    return x_t / total, t_trace, mu_trace, frac_trace


def grow_reportlogOD(eig_vecs, eig_vals, c_scale, times):
    """
    calculate x(t)
    :param eig_vecs: eigenvectors
    :param eig_vals: eigenvalues
    :param c_scale: rescaling constants
    """
    logOD_trace = np.zeros(len(times))

    for ind_time, time_curr in enumerate(times):
        logOD_trace[ind_time] = np.log(np.sum(np.dot(np.multiply(c_scale, eig_vecs), np.exp(eig_vals * time_curr))))

    return logOD_trace


def generate_env_seq(b, min_sim_time=100, avg_env_length=[10] * 10, seed=False, random_times=True):
    """
    generates a sequence of environments based on transition probabilities and a sequence of exponentially distributed
    environment times
    :param b:
    :param min_sim_time:
    :param avg_env_length:
    :param seed:
    :return: environment sequence and times
    """
    if seed:
        np.random.seed(seed)

    eig_val_b, eig_vec_b = np.linalg.eig(b)
    eig_vec_p = eig_vec_b[:, np.isclose(1, eig_val_b.real)].real
    p = (eig_vec_p / np.sum(eig_vec_p)).flatten()

    env_seq = [np.random.choice(np.shape(b)[0], p=p)]
    if random_times:
        env_times = [np.random.exponential(avg_env_length[env_seq[-1]])]
    else:
        env_times = [avg_env_length[env_seq[-1]]]
    sum_env_times = env_times[0]
    ind = 0

    while sum_env_times < min_sim_time:
        env_seq.append(np.random.choice(np.shape(b)[0], p=b[:, env_seq[ind]]))
        if random_times:
            env_times.append(np.random.exponential(avg_env_length[env_seq[-1]]))
        else:
            env_times.append(avg_env_length[env_seq[-1]])
        sum_env_times += env_times[-1]
        ind += 1

    return env_seq, env_times


def calc_eig_vals_vecs_approximations(mat, fit_mat):
    num_phen = mat.shape[0]
    mus = np.diag(fit_mat)
    zero_vals = mus.copy()
    zero_vecs = np.zeros(mat.shape)
    np.fill_diagonal(zero_vecs, 1)

    delta_h = mat.copy() - fit_mat
    h_mat = mat.copy()
    np.fill_diagonal(h_mat, 0)

    mu_diffs = np.tile(mus, (num_phen, 1)) - np.transpose(np.tile(mus, (num_phen, 1)))
    np.fill_diagonal(mu_diffs, 1)
    h_mat_rescale = np.divide(h_mat, mu_diffs)

    # Calculate first order approximation of eigenvalues
    first_val_corr = delta_h.diagonal()
    first_vals = zero_vals + first_val_corr

    # Calculate first order approximation of eigenvectors
    first_vecs_corr = h_mat_rescale.copy()
    first_vecs = zero_vecs + first_vecs_corr

    # Calculate second order approximation of eigenvalues
    second_val_corr = np.diagonal(np.dot(h_mat, h_mat_rescale))
    second_vals = first_vals + second_val_corr

    # Calculate second order approximation of eigenvectors
    switching_backforth = np.matmul(h_mat, np.divide(h_mat, mu_diffs))
    HiiminusHkk = np.transpose(np.tile(delta_h.diagonal(), (num_phen, 1))) - np.tile(delta_h.diagonal(),
                                                                                     (num_phen, 1))
    switched_i_to_k = np.divide(h_mat, mu_diffs)
    second_vecs_corr = np.divide(switching_backforth + np.multiply(HiiminusHkk, switched_i_to_k), mu_diffs)

    np.fill_diagonal(second_vecs_corr, - 0.5 * np.sum(np.multiply(h_mat_rescale, h_mat_rescale), axis=0))
    second_vecs = first_vecs + second_vecs_corr

    # Do some tests:
    zero_vals_matrix = np.zeros(fit_mat.shape)
    np.fill_diagonal(zero_vals_matrix, zero_vals)
    # Does the zeroth order perturbation equation work out?
    test_zero = np.dot(fit_mat, zero_vecs) - np.dot(zero_vals_matrix, zero_vecs)

    # Does the first order perturbation equation work out?
    first_vals_matrix = np.zeros(fit_mat.shape)
    np.fill_diagonal(first_vals_matrix, first_val_corr)
    test_first = np.matmul(fit_mat, first_vecs_corr) + np.matmul(delta_h, zero_vecs) - np.matmul(first_vecs_corr,
                                                                                                 zero_vals_matrix) - np.matmul(
        first_vals_matrix, zero_vecs)

    # Does the second order perturbation equation work out?
    second_vals_matrix = np.zeros(fit_mat.shape)
    np.fill_diagonal(second_vals_matrix, second_val_corr)
    test_second = np.matmul(fit_mat, second_vecs_corr) + np.matmul(delta_h, first_vecs_corr) - np.matmul(
        second_vecs_corr, zero_vals_matrix) - np.matmul(first_vecs_corr, first_vals_matrix) - \
                  np.matmul(zero_vecs, second_vals_matrix)

    # Calculate inverses
    # Zeroth order
    zero_inv = np.zeros(mat.shape)
    np.fill_diagonal(zero_inv, 1)

    # First order
    first_inv_corr = - np.divide(h_mat, mu_diffs)
    first_inv = zero_inv + first_inv_corr

    # Second order
    # First fill the off-diagonal elements
    switching_ik_via_l = np.matmul(np.divide(h_mat, mu_diffs), np.divide(h_mat, mu_diffs))
    switching_il_lk = np.divide(np.matmul(h_mat, np.divide(h_mat, mu_diffs)), mu_diffs)
    switching_away_inv = np.multiply(HiiminusHkk, np.divide(h_mat, np.multiply(mu_diffs, mu_diffs)))
    second_inv_corr = switching_ik_via_l - switching_il_lk - switching_away_inv

    np.fill_diagonal(second_inv_corr, np.sum(
        np.divide(np.multiply(h_mat, 0.5 * h_mat - np.transpose(h_mat)), np.multiply(mu_diffs, mu_diffs)), axis=0))
    second_inv = first_inv + second_inv_corr

    # Test inverses
    test_inv_zero = np.matmul(zero_vecs, zero_inv)
    test_inv_first = np.matmul(first_vecs_corr, zero_inv) + np.matmul(zero_vecs, first_inv_corr)
    test_inv_second = np.matmul(zero_vecs, second_inv_corr) + np.matmul(first_vecs_corr, first_inv_corr) + np.matmul(
        second_vecs_corr, zero_inv)

    return zero_vals, first_vals, second_vals, zero_vecs, first_vecs, second_vecs, zero_inv, first_inv, second_inv


def calc_q_ij(mats, fit_mats, dominant_eigs_indices, order='second'):
    num_env = mats.shape[0]
    num_phen = mats.shape[1]
    q_ij = np.zeros((num_env, num_env))

    for env_i in range(num_env):
        # Collect all necessary matrices
        fit_mat_i = fit_mats[env_i]
        mus_i = np.diag(fit_mat_i)
        h_mat_i = mats[env_i].copy()
        np.fill_diagonal(h_mat_i, 0)
        mu_diffs_i = np.tile(mus_i, (num_phen, 1)) - np.transpose(np.tile(mus_i, (num_phen, 1)))
        np.fill_diagonal(mu_diffs_i, 1)
        alpha_i = dominant_eigs_indices[env_i]
        HcolcolminusHrowrow_i = np.tile(np.sum(h_mat_i,axis=0),(num_phen,1)) - np.transpose(np.tile(np.sum(h_mat_i,axis=0), (num_phen, 1)))

        for env_j in range(num_env):
            # Collect all necessary matrices
            fit_mat_j = fit_mats[env_j]
            mus_j = np.diag(fit_mat_j)
            h_mat_j = mats[env_j].copy()
            np.fill_diagonal(h_mat_j, 0)
            mu_diffs_j = np.tile(mus_j, (num_phen, 1)) - np.transpose(np.tile(mus_j, (num_phen, 1)))
            np.fill_diagonal(mu_diffs_j, 1)
            alpha_j = dominant_eigs_indices[env_j]
            HcolcolminusHrowrow_j = np.tile(np.sum(h_mat_j, axis=0), (num_phen, 1)) - np.transpose(
                np.tile(np.sum(h_mat_j, axis=0), (num_phen, 1)))

            if alpha_i == alpha_j:  # We have a different formula if the dominant eigenvector remains at same index
                q_curr = 1

                q_first_corr = 0
                q_curr = q_curr + q_first_corr
                if order == 'second':
                    # first order inverse times first order vectors
                    first_first_corr = - np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_j, mu_diffs_j))[alpha_i, alpha_j]

                    # second order inverse times zeroth order vectors
                    second_zero_corr = 0.5 * np.sum(np.multiply(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_i, mu_diffs_i))[:, alpha_i])
                    second_zero_corr = second_zero_corr - np.matmul(h_mat_i, np.divide(h_mat_i, np.multiply(mu_diffs_i, mu_diffs_i)))[alpha_i, alpha_i]

                    # zeroth order inverse times second order vectors
                    zero_second_corr = - 0.5 * np.sum(np.multiply(np.divide(h_mat_j, mu_diffs_j), np.divide(h_mat_j, mu_diffs_j))[:, alpha_j])
                    q_second_corr = first_first_corr + second_zero_corr + zero_second_corr
                    q_curr = q_curr + q_second_corr

            else:  # If the index of dominant eigenvector changes
                q_curr = 0

                q_first_corr = (np.divide(h_mat_j, mu_diffs_j) - np.divide(h_mat_i, mu_diffs_i))[alpha_i,alpha_j]
                q_curr = q_curr + q_first_corr
                if order == 'second':
                    # first order inverse times first order vectors
                    first_first_corr = -np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_j, mu_diffs_j))[alpha_i, alpha_j]

                    # second order inverse times zeroth order vectors
                    second_zero_corr = np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_i, mu_diffs_i))[alpha_i, alpha_j]
                    second_zero_corr = second_zero_corr - np.divide(np.matmul(h_mat_i, np.divide(h_mat_i, mu_diffs_i)),mu_diffs_i)[alpha_i, alpha_j]
                    second_zero_corr = second_zero_corr - np.multiply(HcolcolminusHrowrow_i, np.divide(h_mat_i,np.multiply(mu_diffs_i, mu_diffs_i)))[alpha_i, alpha_j]

                    # zeroth order inverse times second order vectors
                    zero_second_corr = np.divide(np.matmul(h_mat_j, np.divide(h_mat_j, mu_diffs_j)), mu_diffs_j)[alpha_i, alpha_j]
                    zero_second_corr = zero_second_corr + np.multiply(HcolcolminusHrowrow_j,np.divide(h_mat_j, np.multiply(mu_diffs_j, mu_diffs_j)))[alpha_i, alpha_j]
                    q_second_corr = second_zero_corr + first_first_corr + zero_second_corr
                    q_curr = q_curr + q_second_corr

            q_ij[env_i, env_j] = q_curr

    return q_ij


def test_q_ij(mats, fit_mats, dominant_eigs_indices, zero_vecs, zero_inv, first_vecs, first_inv, second_vecs, second_inv):
    order = 'second'

    first_vecs_corr = first_vecs-zero_vecs
    second_vecs_corr = second_vecs - first_vecs
    first_inv_corr = first_inv - zero_inv
    second_inv_corr = second_inv - first_inv

    num_env = mats.shape[0]
    num_phen = mats.shape[1]
    q_ij = np.zeros((num_env, num_env))
    q_ij_test_first = np.zeros((num_env, num_env))
    q_ij_test_second = np.zeros((num_env, num_env))

    for env_i in range(num_env):
        # Collect all necessary matrices
        fit_mat_i = fit_mats[env_i]
        mus_i = np.diag(fit_mat_i)
        h_mat_i = mats[env_i].copy()
        np.fill_diagonal(h_mat_i, 0)
        mu_diffs_i = np.tile(mus_i, (num_phen, 1)) - np.transpose(np.tile(mus_i, (num_phen, 1)))
        np.fill_diagonal(mu_diffs_i, 1)
        alpha_i = dominant_eigs_indices[env_i]
        HcolcolminusHrowrow_i = np.tile(np.sum(h_mat_i,axis=0),(num_phen,1)) - np.transpose(np.tile(np.sum(h_mat_i,axis=0), (num_phen, 1)))

        for env_j in range(num_env):
            # Collect all necessary matrices
            fit_mat_j = fit_mats[env_j]
            mus_j = np.diag(fit_mat_j)
            h_mat_j = mats[env_j].copy()
            np.fill_diagonal(h_mat_j, 0)
            mu_diffs_j = np.tile(mus_j, (num_phen, 1)) - np.transpose(np.tile(mus_j, (num_phen, 1)))
            np.fill_diagonal(mu_diffs_j, 1)
            alpha_j = dominant_eigs_indices[env_j]
            HcolcolminusHrowrow_j = np.tile(np.sum(h_mat_j, axis=0), (num_phen, 1)) - np.transpose(
                np.tile(np.sum(h_mat_j, axis=0), (num_phen, 1)))

            q_ij_supposed_first = (np.matmul(zero_inv[env_i], first_vecs_corr[env_j]) + np.matmul(first_inv_corr[env_i], zero_vecs[env_j]))[alpha_i, alpha_j]
            q_ij_supposed_second = (np.matmul(zero_inv[env_i], second_vecs_corr[env_j]) + np.matmul(first_inv_corr[env_i], first_vecs_corr[env_j]) + np.matmul(second_inv_corr[env_i], zero_vecs[env_j]))[alpha_i,alpha_j]

            if alpha_i == alpha_j:  # We have a different formula if the dominant eigenvector remains at same index
                q_curr = 1

                supposed_firstcorr = (np.matmul(zero_inv[env_i], first_vecs_corr[env_j]) + np.matmul(first_inv_corr[env_i], zero_vecs[env_j]))[alpha_i, alpha_j]

                q_first_corr = 0
                q_curr = q_curr + q_first_corr
                if order == 'second':
                    # first order inverse times first order vectors
                    supposed_firstfirst_corr = np.matmul(first_inv_corr[env_i], first_vecs_corr[env_j])[alpha_i, alpha_j]

                    first_first_corr = - np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_j, mu_diffs_j))[alpha_i, alpha_j]

                    # second order inverse times zeroth order vectors
                    supposed_secondzero_corr = np.matmul(second_inv_corr[env_i], zero_vecs[env_j])[alpha_i, alpha_j]

                    second_zero_corr = 0.5 * np.sum(np.multiply(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_i, mu_diffs_i))[:, alpha_i])
                    second_zero_corr = second_zero_corr - np.matmul(h_mat_i, np.divide(h_mat_i, np.multiply(mu_diffs_i, mu_diffs_i)))[alpha_i, alpha_i]

                    # zeroth order inverse times second order vectors
                    supposed_zerosecond_corr = np.matmul(zero_inv[env_i], second_vecs_corr[env_j])[alpha_i, alpha_j]

                    zero_second_corr = - 0.5 * np.sum(np.multiply(np.divide(h_mat_j, mu_diffs_j), np.divide(h_mat_j, mu_diffs_j))[:, alpha_j])
                    q_second_corr = first_first_corr + second_zero_corr + zero_second_corr
                    q_curr = q_curr + q_second_corr

            else:  # If the index of dominant eigenvector changes
                q_curr = 0
                supposed_firstcorr = (np.matmul(zero_inv[env_i], first_vecs_corr[env_j]) + np.matmul(first_inv_corr[env_i], zero_vecs[env_j]))[alpha_i, alpha_j]

                q_first_corr = (np.divide(h_mat_j, mu_diffs_j) - np.divide(h_mat_i, mu_diffs_i))[alpha_i,alpha_j]
                q_curr = q_curr + q_first_corr
                if order == 'second':
                    # first order inverse times first order vectors
                    supposed_firstfirst_corr = np.matmul(first_inv_corr[env_i], first_vecs_corr[env_j])[alpha_i, alpha_j]

                    first_first_corr = -np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_j, mu_diffs_j))[alpha_i, alpha_j]

                    # second order inverse times zeroth order vectors
                    supposed_secondzero_corr = np.matmul(second_inv_corr[env_i], zero_vecs[env_j])[alpha_i, alpha_j]

                    second_zero_corr = np.matmul(np.divide(h_mat_i, mu_diffs_i), np.divide(h_mat_i, mu_diffs_i))[alpha_i, alpha_j]
                    second_zero_corr = second_zero_corr - np.divide(np.matmul(h_mat_i, np.divide(h_mat_i, mu_diffs_i)),mu_diffs_i)[alpha_i, alpha_j]
                    second_zero_corr = second_zero_corr - np.multiply(HcolcolminusHrowrow_i, np.divide(h_mat_i,np.multiply(mu_diffs_i, mu_diffs_i)))[alpha_i, alpha_j]

                    # zeroth order inverse times second order vectors
                    supposed_zerosecond_corr = np.matmul(zero_inv[env_i], second_vecs_corr[env_j])[alpha_i, alpha_j]

                    zero_second_corr = np.divide(np.matmul(h_mat_j, np.divide(h_mat_j, mu_diffs_j)), mu_diffs_j)[alpha_i, alpha_j]
                    zero_second_corr = zero_second_corr + np.multiply(HcolcolminusHrowrow_j,np.divide(h_mat_j, np.multiply(mu_diffs_j, mu_diffs_j)))[alpha_i, alpha_j]
                    q_second_corr = second_zero_corr + first_first_corr + zero_second_corr
                    q_curr = q_curr + q_second_corr

            q_ij[env_i, env_j] = q_curr
            q_ij_test_first[env_i, env_j] = q_ij_supposed_first - q_first_corr
            q_ij_test_second[env_i,env_j] = q_ij_supposed_second - q_second_corr

    return q_ij_test_first, q_ij_test_second


def calc_approx_fitness(mats, fit_mats, avg_wait_times, trans_mat, order='second'):
    num_env = mats.shape[0]
    num_phen = mats.shape[1]

    if np.isscalar(avg_wait_times):  # In this case, all environments have the same avg length
        avg_wait_times = np.ones(num_env)*avg_wait_times

    dominant_eig_inds = np.zeros(num_env)
    dominant_eig_vals = np.zeros(num_env)

    # Store eigenvalues and eigenvector indices (alpha_i) for all environments
    for env_i in range(num_env):
        zero_vals, first_vals, second_vals, zero_vecs, first_vecs, second_vecs, zero_inv, first_inv, second_inv \
            = calc_eig_vals_vecs_approximations(mats[env_i], fit_mats[env_i])
        if order == 'first':
            eig_vals = first_vals
        elif order == 'second':
            eig_vals = second_vals

        dominant_eig_vals[env_i] = np.max(eig_vals)
        dominant_eig_inds[env_i] = np.argmax(eig_vals)

    dominant_eig_inds = dominant_eig_inds.astype(int)

    # Get transition losses (q_ij)
    q_ij = calc_q_ij(mats, fit_mats, dominant_eig_inds, order=order)

    # Get probabilities of environments as right eigenvector with eigenvalue 1
    eig_val_b, eig_vec_b = np.linalg.eig(trans_mat)
    eig_vec_p = eig_vec_b[:, np.isclose(1, eig_val_b.real)].real
    p = (eig_vec_p / np.sum(eig_vec_p)).flatten()

    # Calculate average duration of environments
    tau = np.dot(p, avg_wait_times)

    # Add everything up to get average fitness
    # First the optimal growth (in the limiting case that population is always in dominant eigenvector immediately)
    max_growth = np.sum(np.multiply(np.multiply(p, avg_wait_times), dominant_eig_vals))
    # Then account for the cells that are lost during transitions
    transition_loss = np.sum(np.matmul(np.multiply(trans_mat, np.log(q_ij)), p))

    avg_fitness = (max_growth + transition_loss)/tau

    return avg_fitness



