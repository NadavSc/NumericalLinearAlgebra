import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from time import time


def error_estimation_algorithm1(A, U2, B1, tau_eps=1e-1):
    """First error estimation algorithm using submatrices"""
    N = A.shape[0]
    l = 1
    n = 5
    eps_prev = None

    while True:
        # Step 2: Draw random indices
        irow = np.random.choice(N, size=n, replace=False)
        icol = np.random.choice(N, size=n, replace=False)

        # Step 3: Extract submatrix
        A_l = A[np.ix_(irow, icol)]

        # Step 4: Compute approximated submatrix
        A_tilde_l = U2[irow, :] @ B1[:, icol]

        # Step 5: Compute relative error
        eps_l = np.linalg.norm(A_l - A_tilde_l, 2) / np.linalg.norm(A_l, 2)

        # Step 6: Check convergence
        if l != 1:
            e_l = abs(eps_l - eps_prev) / eps_l
            if e_l < tau_eps:
                return eps_l

        eps_prev = eps_l
        l += 1
        n = 2 * n

        # Safety check for maximum iterations
        if n > N:
            return eps_l


def error_estimation_algorithm2(A, U2, B1, tau_eps=1e-1):
    """Second error estimation algorithm using diagonal elements"""
    N = A.shape[0]
    l = 1
    n = 5
    eps_prev = None

    while True:
        # Step 2: Draw random indices
        irow = np.random.choice(N, size=n, replace=False)
        icol = np.random.choice(N, size=n, replace=False)

        # Step 3-4: Build vectors
        a_l = np.array([A[irow[j], icol[j]] for j in range(n)])
        a_tilde_l = np.array([U2[irow[j], :] @ B1[:, icol[j]] for j in range(n)])

        # Step 5: Compute relative error
        eps_l = np.linalg.norm(a_l - a_tilde_l, 2) / np.linalg.norm(a_l, 2)

        # Step 6: Check convergence
        if l != 1:
            e_l = abs(eps_l - eps_prev) / eps_l
            if e_l < tau_eps:
                return eps_l

        eps_prev = eps_l
        l += 1
        n = 2 * n

        # Safety check for maximum iterations
        if n > N:
            return eps_l


def low_rank_approximation(A, tau):
    start_time = time()
    U, s, Vh = np.linalg.svd(A)
    k = np.sum(s > tau * s[0])
    A_lr = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
    return A_lr, k, s, time() - start_time


def randomized_lr_approximation(A, B0, tau, term_cond_mode='original'):
    """Implement the randomized LR approximation algorithm"""
    termination_conditions = {'original': lambda B, A_norm_F: np.linalg.norm(B, 'fro'),
                              'alg1': lambda A, U2, B1: error_estimation_algorithm1(A, U2, B1),
                              'alg2': lambda A, U2, B1: error_estimation_algorithm2(A, U2, B1)}
    start_time = time()
    N = A.shape[0]
    U2 = np.array([]).reshape(N, 0)
    B1 = np.array([]).reshape(0, N)
    if term_cond_mode == 'original':
        A_norm_F = np.linalg.norm(A, 'fro')
    l = 1
    A_current = A.copy()

    while True:
        # Step 2: Generate random Gaussian matrix
        G = np.random.normal(0, 1, (N, B0)) + 1j * np.random.normal(0, 1, (N, B0))

        # Step 3: Compute M
        M = A_current @ G

        # Step 4: Compute SVD
        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        # Step 5: Compute B
        B = (U[:, :B0].conj().T @ A_current)

        # Step 6: Update U2 and B1
        U2 = np.hstack([U2, U[:, :B0]])
        B1 = np.vstack([B1, B])

        estimated_error_func = termination_conditions[term_cond_mode]
        if term_cond_mode == 'original':
            estimated_error = estimated_error_func(B, A_norm_F)
        else:
            estimated_error = estimated_error_func(A, U2, B1)
        term_cond = estimated_error <= tau if term_cond_mode != 'original' else estimated_error <= tau * A_norm_F

        # Step 7: Check termination condition
        if term_cond:
            computation_time = time() - start_time
            return U2, B1, l * B0, computation_time, estimated_error

        # Step 8: Update A and l
        A_current = A_current - U[:, :B0] @ B
        l += 1


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


def run_analysis(N=128, lambda_=1, W=128, theta=0, D=128, B0=5):
    """Run analysis for different tau values"""
    delta = lambda_ / 10
    A = construct_A(lambda_, theta, delta, W, D)

    # Test different tau values
    tau_values = np.logspace(-10, -1, num=10)
    computation_time_lr = []
    computation_time_fast_lr = []
    lr_errors = []
    fast_lr_errors = []
    ranks_lr = []
    ranks_fast_lr = []

    for tau in tau_values:
        A_lr, rank_lr, _, comp_time_lr = low_rank_approximation(A, tau)
        U2, B1, rank_fast_lr, comp_time_fast_lr, _ = randomized_lr_approximation(A, B0, tau)

        # Compute actual relative error
        A_fast_lr = U2 @ B1
        lr_error = np.linalg.norm(A - A_lr) / np.linalg.norm(A)
        fast_lr_error = np.linalg.norm(A - A_fast_lr) / np.linalg.norm(A)

        computation_time_lr.append(comp_time_lr)
        computation_time_fast_lr.append(comp_time_fast_lr)
        lr_errors.append(lr_error)
        fast_lr_errors.append(fast_lr_error)
        ranks_lr.append(rank_lr)
        ranks_fast_lr.append(rank_fast_lr)

    return tau_values, computation_time_lr, computation_time_fast_lr, lr_errors, fast_lr_errors, ranks_lr, ranks_fast_lr


if __name__ == '__main__':
    section = 'i'

    if section == 'i':
        tau_values, computation_time_lr, computation_time_fast_lr, lr_errors, fast_lr_errors, ranks_lr, ranks_fast_lr = run_analysis()

        plt.semilogx(tau_values, computation_time_lr, 'o', label='LR Approx')
        plt.semilogx(tau_values, computation_time_fast_lr, 's', label='Fast LR Approx')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('τ')
        plt.ylabel('time (s)')
        plt.legend()
        plt.title('LR Approximation Computation Time')
        plt.savefig('part3_i_computation_time.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        # Plot actual vs target error
        plt.loglog(tau_values, lr_errors, 'o', label='LR Approx')
        plt.loglog(tau_values, fast_lr_errors, 's', label='Fast LR Approx')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('τ')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.title('LR Approximation Relative Error')
        plt.savefig('part3_i_errors.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        # Create a scatter plot of rank vs tau
        plt.semilogx(tau_values, ranks_lr, 'o', label='LR Approx')
        plt.semilogx(tau_values, ranks_fast_lr, 's', label='Fast LR Approx')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.xlabel('τ')
        plt.ylabel('rank')
        plt.title('Rank of LR Approximation Methods')
        plt.savefig('part3_i_ranks.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

    if section == 'j':
        def run_comparison(N=128, lambda_=1, W=128, theta=0, D=128, B0=5):
            """Run comparison between original and new implementations"""
            delta = lambda_ / 10
            A = construct_A(lambda_, theta, delta, W, D)

            tau_values = np.logspace(-10, -1, num=10)
            results = {
                'original': {'times': [], 'errors': [], 'ranks': []},
                'alg1': {'times': [], 'errors': [], 'ranks': []},
                'alg2': {'times': [], 'errors': [], 'ranks': []}
            }

            # Run comparisons
            for tau in tau_values:
                # Original implementation
                U2, B1, rank, time_orig, estimated_error = randomized_lr_approximation(A, B0, tau, 'original')
                error_orig = np.linalg.norm(A - U2 @ B1, 'fro') / np.linalg.norm(A, 'fro')
                results['original']['times'].append(time_orig)
                results['original']['errors'].append(error_orig)
                results['original']['ranks'].append(rank)

                # Algorithm 1
                U2, B1, rank, time_alg1, error_alg1 = randomized_lr_approximation(A, B0, tau, 'alg1')
                results['alg1']['times'].append(time_alg1)
                results['alg1']['errors'].append(error_alg1)
                results['alg1']['ranks'].append(rank)

                # Algorithm 2
                U2, B1, rank, time_alg2, error_alg2 = randomized_lr_approximation(A, B0, tau, 'alg2')
                results['alg2']['times'].append(time_alg2)
                results['alg2']['errors'].append(error_alg2)
                results['alg2']['ranks'].append(rank)

            return tau_values, results


        # Run comparison and plot results
        tau_values, results = run_comparison()

        # Computation Time
        plt.semilogx(tau_values, results['original']['times'], 'o', label='Original')
        plt.semilogx(tau_values, results['alg1']['times'], 's', label='Fast Error Estimation 1')
        plt.semilogx(tau_values, results['alg2']['times'], '^', label='Fast Error Estimation 2')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('τ')
        plt.ylabel('Computation Time (s)')
        plt.legend()
        plt.title('Computation Time Comparison')
        plt.savefig('part3_j_computation_time.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        # Error Comparison
        plt.loglog(tau_values, results['original']['errors'], 'o', label='Original')
        plt.loglog(tau_values, results['alg1']['errors'], 's', label='Fast Error Estimation 1')
        plt.loglog(tau_values, results['alg2']['errors'], '^', label='Fast Error Estimation 2')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('τ')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.title('Error Comparison')
        plt.savefig('part3_j_errors.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

        # Rank Comparison
        plt.semilogx(tau_values, results['original']['ranks'], 'o', label='Original')
        plt.semilogx(tau_values, results['alg1']['ranks'], 's', label='Fast Error Estimation 1')
        plt.semilogx(tau_values, results['alg2']['ranks'], '^', label='Fast Error Estimation 2')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('τ')
        plt.ylabel('rank')
        plt.title('Rank Comparison')
        plt.legend()
        plt.savefig('part3_j_rank.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()