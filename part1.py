import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
matplotlib.use('TkAgg')


def construct_A(lambda_, W, theta, alpha):
    delta = lambda_/10
    k = 2 * np.pi/lambda_
    N = int(W/delta)
    D = alpha*W

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


def compute_rank(S, threshold):
    return np.sum(S > threshold * S[0])


def compute_SVD(A):
    start_time = time()
    _, s_values, _ = np.linalg.svd(A)
    svd_time = time() - start_time

    ranks = [compute_rank(s_values, tau) for tau in [1e-2, 1e-5, 1e-8]]
    condition_number = s_values[0] / s_values[-1]

    return s_values, svd_time, ranks, condition_number


if __name__ == '__main__':
    lambda_ = 1
    W = 4*lambda_
    theta = np.pi/2
    alpha = 1
    A = construct_A(lambda_=lambda_,
                    W=W,
                    theta=theta,
                    alpha=alpha)

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
    plt.savefig('ex1_a_constructA.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    # Plot Matrix A Singular Values
    s_values, svd_time, ranks, condition_number = compute_SVD(A)
    plt.scatter(np.arange(len(s_values)), s_values)
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid
    plt.title('Singular Values of Matrix A')
    plt.xlabel('Transmitter Index')
    plt.ylabel('Singular Values')
    plt.savefig('ex1_a_svdA_log.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()