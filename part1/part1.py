import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from time import time
from logger import set_logger, info
matplotlib.use('TkAgg')

set_logger(log_path=os.path.join('../logger', 'log.txt'))


def construct_A(lambda_, theta, delta, W, D):
    k = 2 * np.pi/lambda_
    N = int(W/delta)

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


if __name__ == '__main__':
    sections = ['b']  # Possible sections: 'a', 'b', 'c', 'd'
    for section in sections:
        if section == 'a':
            lambda_ = 1
            delta = lambda_ / 10
            theta = np.pi / 2
            W = 4 * lambda_
            alpha = 1
            D = alpha * W
            A = construct_A(lambda_=lambda_,
                            theta=theta,
                            delta=delta,
                            W=W,
                            D=D)
            info('Section A: A has been constructed')

            # Plot Matrix A
            plt.imshow(np.abs(A))
            x_ticks = np.arange(0, len(A)+1, 5)[1:]-1
            y_ticks = np.arange(0, len(A)+1, 5)[1:]-1
            plt.xticks(x_ticks, labels=x_ticks+1)
            plt.yticks(y_ticks, labels=y_ticks+1)
            plt.colorbar()
            plt.title('Absolute Values of Matrix A')
            plt.xlabel('Transmitter Index')
            plt.ylabel('Receiver Index')
            plt.savefig(r'ex1_a_constructA.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

            # Plot Matrix A Singular Values
            s_values, _, _, _ = compute_SVD(A)
            info('Section A: SVD has been calculated')
            plt.scatter(np.arange(len(s_values)), s_values)
            plt.yscale('log')  # Set y-axis to log scale
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid
            plt.title('Singular Values of Matrix A')
            plt.xlabel('Transmitter Index')
            plt.ylabel('Singular Values')
            plt.savefig(r'ex1_a_svdA_log.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

        ## TODO: add proper asymptotic lines to figures
        if section == 'b':
            lambda_ = 1
            delta = lambda_ / 10
            theta = np.pi / 2
            W_values = [4 * lambda_ * (2 ** i) for i in range(7)]
            N_values = [int(w/delta) for w in W_values]

            D_cases = {
                'D = 4λ': lambda W: 4 * lambda_,
                'D = W': lambda W: W,
                'D = W²/λ': lambda W: W ** 2 / lambda_
            }
            if os.path.exists(r'results_b.pkl'):
                with open(r'results_b.pkl', 'rb') as f:
                    results = pickle.load(f)
                info('results_b.pkl has been loaded')
            else:
                results = {case: {'N': [], 'time': [], 'ranks': [], 'condition': []} for case in D_cases}

                for case, D_func in D_cases.items():
                    for N, W in zip(N_values, W_values):
                        D = D_func(W)
                        A = construct_A(lambda_=lambda_,
                                        theta=theta,
                                        delta=delta,
                                        W=W,
                                        D=D)
                        info(f'Section B: A with {case}, W={W}, N={N} has been constructed')
                        s_values, svd_time, ranks, condition_number = compute_SVD(A)
                        info(f'Section B: SVD has been calculated')
                        results[case]['N'].append(N)
                        results[case]['time'].append(svd_time)
                        results[case]['ranks'].append(ranks)
                        results[case]['condition'].append(condition_number)
                with open(r'results_b.pkl', 'wb') as f:
                    pickle.dump(results, f)

            for case, data in results.items():
                plt.loglog(data['N'], data['time'], 'o', label=case, markersize=8)

            # Theoretical complexity O(N^3)
            N_theory = np.logspace(1.5, 3, 100)
            time_theory = 1e-7 * N_theory ** 3  # Adjust the constant for visual fit
            plt.loglog(N_theory, time_theory, 'k--', label='O(N³) trend')

            plt.xlabel('Number of Antennas (N)')
            plt.ylabel('SVD Computation Time (s)')
            plt.title('SVD Computation Complexity')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.savefig('part1_b_svd_complexity.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

            for case, data in results.items():
                plt.semilogx(data['N'], [r[0] for r in data['ranks']], 'o', label=f'{case}, τ=1e-2')
                plt.semilogx(data['N'], [r[1] for r in data['ranks']], 's', label=f'{case}, τ=1e-5')
                plt.semilogx(data['N'], [r[2] for r in data['ranks']], '^', label=f'{case}, τ=1e-8')

            plt.xlabel('Number of Antennas (N)')
            plt.ylabel('Rank')
            plt.title('Matrix Rank for Different Truncation Thresholds')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.savefig('part1_b_rank.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

            for i, (case, data) in enumerate(results.items()):
                plt.loglog(data['N'], data['condition'], 'o', label=['τ=1e-2', 'τ=1e-5', 'τ=1e-8'])
                plt.xlabel('Number of Antennas (N)')
                plt.ylabel('Condition Number')
                plt.title(f'Condition Number of Matrix A, {case}')
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.5)

                plt.savefig(f'part1_b_condition_{i}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.show()
                plt.close()