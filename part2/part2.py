import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

from time import time
#from logger import set_logger, info
matplotlib.use('TkAgg')

#set_logger(log_path=os.path.join('../logger', 'log.txt'))


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
    k = np.sum(s > tau*s[0])
    A_lr = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    U[:,k:] = 0
    s[k:] = 0
    Vh[k:,:] = 0
    return A_lr,U,s,Vh,k


def compute_error(A, A_lr):
    return np.linalg.norm(A - A_lr) / np.linalg.norm(A)


def theoretical_error(s_values, rank):
    return np.sqrt(np.sum(s_values[rank:]**2))


if __name__ == '__main__':

    section = 2

    lambda_ = 1
    W = 128 * lambda_
    theta = 0
    delta = lambda_ / 10
    D = W
    N = int(W / delta) + 1

    # Construct matrix A
    A = construct_A(lambda_, theta, delta, W, D)

    plt.imshow(np.abs(A))
    x_ticks = np.arange(0, len(A) + 1, 200)[1:] - 1
    y_ticks = np.arange(0, len(A) + 1, 200)[1:] - 1
    plt.xticks(x_ticks, labels=x_ticks + 1)
    plt.yticks(y_ticks, labels=y_ticks + 1)
    plt.colorbar()
    plt.title('Absolute Values of Matrix A')
    plt.xlabel('Transmitter Index')
    plt.ylabel('Receiver Index')
    plt.savefig(r'ex2_a_constructA.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    U, s, Vh = np.linalg.svd(A)

    # Compute low-rank approximations
    tau_values = np.logspace(-10, -1, num=10)

    if section == 1:
        ranks = []
        errors = []
        theoretical_errors = []
        times = []

        for tau in tau_values:
            A_lr,U_lr,s_lr,Vh_lr,rank = low_rank_approximation(A, tau)

            calc_time = np.mean(timeit.repeat(lambda: compute_error(A, A_lr), repeat=15, number=1))
            ranks.append(rank)
            errors.append(compute_error(A, A_lr))
            theoretical_errors.append(theoretical_error(s_lr, rank))
            times.append(calc_time)
            #info(f'LR tau={tau} approximation has been calculated')

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
    if section == 2:
        tau_ep = 1e-1

        eps = []
        times_f_1 = []
        err_time_calc = []
        req_n = []

        for tau in tau_values:
            el = [1]
            eps_l = []
            n = [5]
            l = 1
            calc_time = 0
            A_lr,U_lr,s_lr,Vh_lr,rank = low_rank_approximation(A, tau)

            while (tau_ep<el[-1]) and (n[-1] < N):
                i_rows = np.array(random.sample(range(N), n[-1]))
                i_cols = np.array(random.sample(range(N), n[-1]))

                A_l = A[i_rows[:, np.newaxis], i_cols]
                A_lr_sub = A_lr[i_rows[:, np.newaxis], i_cols]

                compute_err = compute_error(A_l, A_lr_sub)
                calc_time += np.mean(timeit.repeat(lambda: compute_error(A_l, A_lr_sub), repeat=15, number=1))
                eps_l.append(compute_err)
                if l>1:
                    el.append(np.abs((eps_l[-1] - eps_l[-2])/eps_l[-1]))
                l = l+1
                n.append(2*n[-1])
            req_n.append(n[-1])
            err_time_calc.append(calc_time)
            eps.append(eps_l[-1])

        #plt.loglog(n[1:-1], el[1:], 'o')
        #plt.xlabel('n')
        #plt.ylabel('el')
        #plt.title('el vs n')
        #plt.grid(True, which="both", ls="-", alpha=0.5)
        #plt.show()
        #plt.close()

        plt.loglog(tau_values, eps, 'o')
        plt.xlabel('tau')
        plt.ylabel('epsilon_l')
        plt.title('tau vs epsilon_l')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()

        plt.loglog(tau_values, err_time_calc, 'o')
        plt.xlabel('tau')
        plt.ylabel('err_time_calc')
        plt.title('tau vs err_time_calc')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()

        plt.loglog(tau_values, req_n, 'o')
        plt.xlabel('tau')
        plt.ylabel('req_n')
        plt.title('tau vs req_n')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()



    # Section f second algo
    if section == 3:
        tau_ep = 1e-1

        eps = []
        times_f_1 = []
        err_time_calc = []
        req_n = []


        # Construct matrix A
        for tau in tau_values:
            el = [1]
            eps_l = []
            n = [5]
            l = 1
            calc_time = 0

            A_lr,U_lr,s_lr,Vh_lr,rank = low_rank_approximation(A, tau)

            Vh_new = np.matmul(np.diag(s_lr), Vh_lr)

            while (tau_ep<el[-1]) and (n[-1] < N):
                i_rows = random.sample(range(N), n[-1])
                i_cols = random.sample(range(N), n[-1])

                a_l = np.array([A[i_rows[j], i_cols[j]] for j in range(n[-1])])
                a_l_sub = np.array([np.dot(U_lr[i_rows[j], :], Vh_new[:, i_cols[j]]) for j in range(n[-1])])

                compute_err = compute_error(a_l, a_l_sub)
                calc_time += np.mean(timeit.repeat(lambda: compute_error(a_l, a_l_sub), repeat=15, number=1))
                eps_l.append(compute_err)

                if l>1:
                    el.append(np.abs((eps_l[-1] - eps_l[-2])/eps_l[-1]))
                l = l+1
                n.append(2 * n[-1])
            req_n.append(n[-1])
            err_time_calc.append(calc_time)
            eps.append(eps_l[-1])

        plt.loglog(tau_values, eps, 'o')
        plt.xlabel('tau')
        plt.ylabel('epsilon_l')
        plt.title('tau vs epsilon_l')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()

        plt.loglog(tau_values, err_time_calc, 'o')
        plt.xlabel('tau')
        plt.ylabel('err_time_calc')
        plt.title('tau vs err_time_calc')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()

        plt.loglog(tau_values, req_n, 'o')
        plt.xlabel('tau')
        plt.ylabel('req_n')
        plt.title('tau vs req_n')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
        plt.close()