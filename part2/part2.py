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


def low_rank_approximation(A, tau):
    U, s, Vh = np.linalg.svd(A)
    k = np.sum(s > tau * s[0])
    A_lr = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    return A_lr, k, s


def compute_error(A, A_lr):
    return np.linalg.norm(A - A_lr) / np.linalg.norm(A)


def theoretical_error(s_values, rank):
    return s_values[rank] / np.linalg.norm(A)


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


def error_estimation_vector(A: np.ndarray, U: np.ndarray, V: np.ndarray,
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

        # Extract elements
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
        rank = np.sum(s >= tau)
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
    results_vector = []

    for tau in tau_values:
        # Exact computation
        start_time = time()
        U, V = low_rank_approx(A, tau)
        A_approx = U @ V
        exact_error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
        exact_time = time() - start_time
        results_exact.append((exact_error, U.shape[1], exact_time))

        # Estimation methods
        eps_sub, n_sub, time_sub = error_estimation_submatrix(A, U, V)
        eps_diag, n_diag, time_diag = error_estimation_vector(A, U, V)

        results_submatrix.append((eps_sub, n_sub, time_sub))
        results_vector.append((eps_diag, n_diag, time_diag))

    return tau_values, results_exact, results_submatrix, results_vector


if __name__ == '__main__':
    lambda_ = 1
    W = 128 * lambda_
    theta = 0
    delta = lambda_ / 10
    D = W
    N = int(W/delta)
    sections = ['f']  # Possible sections: 'e', 'f'
    show = True
    save = False
    if 'e' in sections:
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

            calc_time = np.mean(timeit.repeat(lambda: low_rank_approximation(A, tau), repeat=5, number=1))
            ranks.append(rank)
            errors.append(compute_error(A, A_lr))
            theoretical_errors.append(theoretical_error(s_values, rank))
            times.append(calc_time)
            info(f'LR tau={tau} approximation has been calculated')

        plt.semilogx(tau_values, ranks, 'o')
        plt.xlabel('τ')
        plt.ylabel('Rank')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_e_lr_approx_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        plt.loglog(tau_values, errors, 'o', label='Error')
        plt.loglog(tau_values, theoretical_errors, 'x', label='Theoretical error')
        plt.xlabel('τ')
        plt.ylabel('Relative Error (2-norm)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_e_lr_approx_error.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        plt.semilogx(tau_values, times, 'o')
        plt.xlabel('τ')
        plt.ylabel('Time (s)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_e_lr_approx_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        print(f"τ values: {tau_values}")
        print(f"Ranks: {ranks}")
        print(f"Errors: {errors}")
        print(f"Times: {times}")

    if 'f' in sections:
        # Run analysis and plot results
        tau_values, results_exact, results_submatrix, results_vector = fast_relative_error_estimation()

        # Extract results
        errors_exact = [r[0] for r in results_exact]
        ranks = [r[1] for r in results_exact]
        times_exact = [r[2] for r in results_exact]

        errors_sub = [r[0] for r in results_submatrix]
        ns_sub = [r[1] for r in results_submatrix]
        times_sub = [r[2] for r in results_submatrix]

        errors_diag = [r[0] for r in results_vector]
        ns_diag = [r[1] for r in results_vector]
        times_diag = [r[2] for r in results_vector]

        plt.loglog(tau_values, errors_exact, 'o', label='Original')
        plt.loglog(tau_values, errors_sub, 's', label='FEE1')
        plt.loglog(tau_values, errors_diag, '^', label='FEE2')
        plt.xlabel('τ')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_f_errors.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        plt.semilogx(tau_values, ranks, 'o', label='Rank')
        plt.xlabel('τ')
        plt.ylabel('Rank')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_f_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        plt.semilogx(tau_values, ns_sub, 's', color='C1', label='FEE1')
        plt.semilogx(tau_values, ns_diag, '^', color='C2', label='FEE2')
        plt.xlabel('τ')
        plt.ylabel('n')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_f_final_n.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()

        plt.loglog(tau_values, times_exact, 'o', label='Original')
        plt.loglog(tau_values, times_sub, 's', label='FEE1')
        plt.loglog(tau_values, times_diag, '^', label='FEE2')
        plt.xlabel('τ')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if save:
            plt.savefig('part2_f_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        plt.close()