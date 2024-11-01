import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from typing import Tuple, List
from time import time
from logger import set_logger, info
matplotlib.use('TkAgg')

set_logger(log_path=os.path.join('../logger', 'log.txt'))


def construct_A(lambda_, theta, delta, W, D):
    k = 2 * np.pi/lambda_
    N = int(W/delta) + 1

    r_t = np.zeros((N, 2))
    r_r = np.zeros((N, 2))
    A = np.zeros((N, N), dtype=complex)

    for n in range(N):
        d = (n+1)*delta
        r_t[n] = (D/2 + d*np.cos(theta), d*np.sin(theta))
        r_r[n] = (-D/2 - d*np.cos(theta), d*np.sin(theta))

    for m in range(N):
        for n in range(N):
            distance = np.linalg.norm(r_r[m] - r_t[n])
            A[m, n] = np.exp(-1j * k * distance) / np.sqrt(distance)

    return A


def compute_rank(s_values, threshold):
    return np.sum(s_values > threshold)


def compute_SVD(A):
    start_time = time()
    _, s_values, _ = np.linalg.svd(A)
    svd_time = time() - start_time

    ranks = [compute_rank(s_values, tau) for tau in [1e-2, 1e-5, 1e-8]]
    condition_numbers = s_values[0] / s_values[np.array(ranks)-1]

    return s_values, svd_time, ranks, condition_numbers


def low_rank_approximation(A, tau):
    U, s, Vh = np.linalg.svd(A)
    k = np.sum(s > s[0]*tau)
    A_lr = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    return A_lr, k, s


def compute_error(A, A_lr):
    return np.linalg.norm(A - A_lr) / np.linalg.norm(A)


def theoretical_error(s_values, rank):
    #return np.sqrt(np.sum(s_values[rank+1:]**2))
    return s_values[rank]/s_values[0]


def error_estimation_submatrix(A: np.ndarray, U: np.ndarray, V: np.ndarray,
                               tau_eps: float = 0.1) -> Tuple[float, int, float]:
    """Implementation of algorithm (£)"""
    start_time = time()
    l = 1
    n = 5
    N = A.shape[0]
    eps_prev = None

    while True:
        # Generate random indices
        i_row = np.random.choice(N, size=n, replace=False)
        i_col = np.random.choice(N, size=n, replace=False)

        # Extract submatrices
        A_l = A[np.ix_(i_row, i_col)]
        A_l_approx = U[i_row, :] @ V[:, i_col]

        # Compute relative error
        eps_l = np.linalg.norm(A_l - A_l_approx) / np.linalg.norm(A_l)

        if l != 1:
            e_l = abs(eps_l - eps_prev) / eps_l
            if e_l < tau_eps:
                return eps_l, n, time() - start_time

        eps_prev = eps_l
        l += 1
        n *= 2

        if n > N:  # Safety check
            return eps_l, n // 2, time() - start_time


def error_estimation_diagonal(A: np.ndarray, U: np.ndarray, V: np.ndarray,
                              tau_eps: float = 0.1) -> Tuple[float, int, float]:
    """Implementation of algorithm (££)"""
    start_time = time()
    l = 1
    n = 5
    N = A.shape[0]
    eps_prev = None

    while True:
        # Generate random indices
        i_row = np.random.choice(N, size=n, replace=False)
        i_col = np.random.choice(N, size=n, replace=False)

        # Extract diagonal elements
        a_l = np.array([A[i_row[j], i_col[j]] for j in range(n)])
        a_l_approx = np.array([U[i_row[j], :] @ V[:, i_col[j]] for j in range(n)])

        # Compute relative error
        eps_l = np.linalg.norm(a_l - a_l_approx) / np.linalg.norm(a_l)

        if l != 1:
            e_l = abs(eps_l - eps_prev) / eps_l
            if e_l < tau_eps:
                return eps_l, n, time() - start_time

        eps_prev = eps_l
        l += 1
        n *= 2

        if n > N:  # Safety check
            return eps_l, n // 2, time() - start_time


# Main analysis
def fast_relative_error_estimation():
    def low_rank_approx(A: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute low-rank approximation using SVD."""
        U, s, Vh = np.linalg.svd(A)
        rank = np.sum(s >= s[0]*tau)
        return U[:, :rank], (s[:rank, None] * Vh[:rank, :])
    # Parameters
    W = 128  # W = 128λ
    theta = 0
    D = W
    N = 128  # Number of antennas
    tau_values = np.logspace(-10, -1, 10)

    # Generate antenna positions and interaction matrix
    A = construct_A(lambda_, theta, delta, W, D)

    # Results storage
    results_exact = []
    results_submatrix = []
    results_diagonal = []
    toatl_n = []

    for tau in tau_values:
        # Exact computation
        U, V = low_rank_approx(A, tau)
        A_approx = U @ V
        # Measuring error calc. time:
        start_time = time()
        exact_error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
        exact_time = time() - start_time
        results_exact.append((exact_error, U.shape[1], exact_time))

        # Estimation methods
        eps_sub, _, _ = error_estimation_submatrix(A, U, V)
        eps_diag, _, _ = error_estimation_diagonal(A, U, V)

        time_sub = np.mean(timeit.repeat(lambda: error_estimation_submatrix(A, U, V), repeat=1000, number=1))
        time_diag = np.mean(timeit.repeat(lambda: error_estimation_diagonal(A, U, V), repeat=1000, number=1))

        n_sub_temp_toatal = []
        n_diag_temp_toatal = []
        for _ in range(1000):
            _, n_sub_temp, _ = error_estimation_submatrix(A, U, V)
            _, n_diag_temp, _ = error_estimation_diagonal(A, U, V)
            n_sub_temp_toatal.append(n_sub_temp)
            n_diag_temp_toatal.append(n_diag_temp)
        n_sub= np.mean(n_sub_temp_toatal)
        n_diag = np.mean(n_diag_temp_toatal)

        results_submatrix.append((eps_sub, n_sub, time_sub))
        results_diagonal.append((eps_diag, n_diag, time_diag))

    return tau_values, results_exact, results_submatrix, results_diagonal


if __name__ == '__main__':
    lambda_ = 1
    W = 128 * lambda_
    theta = 0
    delta = lambda_ / 10
    D = W
    N = int(W/delta)

    section = 'e'

    if section == 'e':
        # Construct matrix A
        A = construct_A(lambda_, theta, delta, W, D)

        # Compute low-rank approximations
        tau_values = np.logspace(-10, -1, num=10)

        ranks = []
        errors = []
        theoretical_errors = []
        times = []

        for tau in tau_values:
            A_lr, rank, s_values = low_rank_approximation(A, tau)

            calc_time = np.mean(timeit.repeat(lambda: compute_error(A, A_lr), repeat=100, number=1))
            ranks.append(rank)
            errors.append(compute_error(A, A_lr))
            theoretical_errors.append(theoretical_error(s_values, rank))
            times.append(calc_time)
            info(f'LR tau={tau} approximation has been calculated')

        plt.imshow(np.abs(A))
        x_ticks = np.arange(0, len(A) + 1, 100)[1:] - 1
        y_ticks = np.arange(0, len(A) + 1, 100)[1:] - 1
        #plt.grid()
        plt.xticks(x_ticks, labels=x_ticks + 1)
        plt.yticks(y_ticks, labels=y_ticks + 1)
        plt.colorbar()
        plt.title('Absolute Values of Matrix A')
        plt.xlabel('Transmitter Index')
        plt.ylabel('Receiver Index')
        plt.savefig(r'ex1_a_constructA.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

        U, s, Vh = np.linalg.svd(A)
        plt.semilogy(np.arange(len(s)), s, 'o')
        plt.xlabel('Transmitter Index')
        plt.ylabel('Singular Values')
        plt.title('Singular Values of A')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('s_vals.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.semilogx(tau_values, ranks, 'o')
        plt.xlabel('τ')
        plt.ylabel('Rank')
        plt.title('Rank of LR Approximation')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('lr_approx_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.loglog(tau_values, errors, 'o', label='Error')
        plt.loglog(tau_values, theoretical_errors, 'x', label='Theoretical error')
        plt.xlabel('τ')
        plt.ylabel('Relative Error (2-norm)')
        plt.title('Relative Error of LR Approximation')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('lr_approx_error.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.semilogx(tau_values, times, 'o')
        plt.xlabel('τ')
        plt.ylabel('Computation Time (s)')
        plt.title('Computation Time for Relative Error Calc.')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('err_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        print(f"τ values: {tau_values}")
        print(f"Ranks: {ranks}")
        print(f"Errors: {errors}")
        print(f"Theoretical Errors: {theoretical_errors}")
        print(f"Times: {times}")
        print(f"First singular values: {s[:10]}")

    if section == 'f':
        # Run analysis and plot results
        tau_values, results_exact, results_submatrix, results_diagonal = fast_relative_error_estimation()

        # Extract results
        errors_exact = [r[0] for r in results_exact]
        ranks = [r[1] for r in results_exact]
        times_exact = [r[2] for r in results_exact]

        errors_sub = [r[0] for r in results_submatrix]
        ns_sub = [r[1] for r in results_submatrix]
        times_sub = [r[2] for r in results_submatrix]

        errors_diag = [r[0] for r in results_diagonal]
        ns_diag = [r[1] for r in results_diagonal]
        times_diag = [r[2] for r in results_diagonal]

        plt.loglog(tau_values, errors_exact, 'o', label='Original')
        plt.loglog(tau_values, errors_sub, 's', label='Submatrix')
        plt.loglog(tau_values, errors_diag, '^', label='Diagonal')
        plt.xlabel('τ')
        plt.ylabel('Relative Error')
        plt.title('Fast Relative Error Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('part2_f_errors.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.semilogx(tau_values, ranks, 'o', label='Rank')
        plt.xlabel('τ')
        plt.ylabel('Rank')
        plt.title('Original Ranks')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('part2_f_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.semilogx(tau_values, ns_sub, 's', label='Submatrix')
        plt.semilogx(tau_values, ns_diag, '^', label='Diagonal')
        plt.xlabel('τ')
        plt.ylabel('n')
        plt.title('Final n')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('part2_f_final_n.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.loglog(tau_values, times_exact, 'o', label='Original')
        plt.loglog(tau_values, times_sub, 's', label='Submatrix')
        plt.loglog(tau_values, times_diag, '^', label='Diagonal')
        plt.xlabel('τ')
        plt.ylabel('time (s)')
        plt.title('Computation Time Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('part2_f_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

        print(f"Times times_exact: {times_exact}")
        print(f"Times times_sub: {times_sub}")
        print(f"Times times_diag: {times_diag}")
