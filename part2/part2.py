import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

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
    k = np.sum(s > tau)
    A_lr = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    return A_lr, k, s


def compute_error(A, A_lr):
    return np.linalg.norm(A - A_lr) / np.linalg.norm(A)


def theoretical_error(s_values, rank):
    return np.sqrt(np.sum(s_values[rank:]**2))


if __name__ == '__main__':

    section = 3

    if section == 1:
        lambda_ = 1
        W = 128 * lambda_
        theta = 0
        delta = lambda_ / 10
        D = W
        N = int(W/delta)

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

            calc_time = np.mean(timeit.repeat(lambda: low_rank_approximation(A, tau), repeat=15, number=1))
            ranks.append(rank)
            errors.append(compute_error(A, A_lr))
            theoretical_errors.append(theoretical_error(s_values, rank))
            times.append(calc_time)
            info(f'LR tau={tau} approximation has been calculated')

        plt.semilogx(tau_values, ranks, 'o')
        plt.xlabel('τ')
        plt.ylabel('Rank')
        plt.title('Rank of LR Approximation')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('lr_approx_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.loglog(tau_values, errors, 'o', label='Error')
        plt.loglog(tau_values, theoretical_errors, 'o', label='Theoretical error')
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
        plt.title('Computation Time for LR Approximation')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('lr_approx_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        print(f"τ values: {tau_values}")
        print(f"Ranks: {ranks}")
        print(f"Errors: {errors}")
        print(f"Times: {times}")


    # Section f first algo
    if section ==2:
        lambda_ = 1
        W = 128 * lambda_
        theta = 0
        delta = lambda_ / 10
        D = W
        N = int(W / delta)
        l = 1
        n = [5]
        tau_ep = 1e-2
        el = [1]
        eps_l = []
        #TODO: Replace tau with array like in (e)
        tau = 1e-1

        # Construct matrix A
        A = construct_A(lambda_, theta, delta, W, D)
        A_lr, rank, lr_s_values = low_rank_approximation(A, tau)

        while tau_ep<el[-1]:
            i_rows = np.array(random.sample(range(N), n[-1]))
            i_cols = np.array(random.sample(range(N), n[-1]))

            A_l = A[i_rows[:, np.newaxis], i_cols]

            #U, s, Vh = np.linalg.svd(A)
            #Vh_new = np.matmul(np.diag(s),Vh)
            #A_l_sub = np.matmul(U[i_rows, :],np.transpose(Vh_new[i_cols, :]))

            A_lr_sub = A_lr[i_rows[:, np.newaxis], i_cols]

            eps_l.append(np.linalg.norm(A_l - A_lr_sub) / np.linalg.norm(A_l))
            if l>1:
                el.append(np.abs((eps_l[-1] - eps_l[-2])/eps_l[-1]))
            l = l+1
            n.append(2*n[-1])

        plt.loglog(n[1:-1], el[1:], 'o')
        plt.xlabel('n')
        plt.ylabel('el')
        plt.title('el vs n')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()

    # Section f second algo
    if section == 3:
        lambda_ = 1
        W = 128 * lambda_
        theta = 0
        delta = lambda_ / 10
        D = W
        N = int(W / delta)
        l = 1
        n = [5]
        tau_ep = 1e-1
        el = [1]
        eps_l = []

        # Construct matrix A
        A = construct_A(lambda_, theta, delta, W, D)

        while tau_ep<el[-1]:
            i_rows = np.array(random.sample(range(N), n[-1]))
            i_cols = np.array(random.sample(range(N), n[-1]))

            a_l = A[i_rows, i_cols]

            U, s, Vh = np.linalg.svd(A)
            Vh_new = np.matmul(np.diag(s),Vh)

            a_l_sub = np.matmul(U[i_rows, :],np.transpose(Vh_new[i_cols, :]))

            eps_l.append(np.linalg.norm(a_l - a_l_sub) / np.linalg.norm(a_l))
            if l>1:
                el.append(np.abs((eps_l[-1] - eps_l[-2])/eps_l[-1]))
            l = l+1
            n.append(2*n[-1])

        plt.loglog(n[1:-1], el[1:], 'o')
        plt.xlabel('n')
        plt.ylabel('el')
        plt.title('el vs n')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()