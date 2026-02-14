import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

def default_config():
    np.random.seed(64)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })

def main():
    p_0: int       = 10
    p_inf: int     = 1
    lota: float    = 0.1
    delta_t: float = 1
    T: int         = 100
    wc: float      = np.exp(-lota * delta_t)
    w1: ndarray    = wc * np.ones(T, dtype=np.float32)
    w2: ndarray    = np.random.normal(loc=wc, scale=0.1, size=[T, ])
    w3: ndarray    = np.random.normal(loc=wc, scale=0.2, size=[T, ])
    w4: ndarray    = np.random.normal(loc=wc, scale=0.3, size=[T, ])

    t: ndarray     = delta_t * np.arange(0, T)
    p: ndarray     = (p_0 - p_inf) * np.exp(-lota * t) + p_inf

    # p_dec_curve_1
    p_dec1: ndarray = p_0 * np.ones(T, dtype=np.float32)
    for i in range(1, T): p_dec1[i] = w1[i] * p_dec1[i-1] + (1 - w1[i]) * p_inf

    # p_dec_curve_2
    p_dec2: ndarray = p_0 * np.ones(T, dtype=np.float32)
    for i in range(1, T): p_dec2[i] = w2[i] * p_dec2[i-1] + (1 - w2[i]) * p_inf

    # p_dec_curve_3
    p_dec3: ndarray = p_0 * np.ones(T, dtype=np.float32)
    for i in range(1, T): p_dec3[i] = w3[i] * p_dec3[i-1] + (1 - w3[i]) * p_inf

    # p_dec_curve_4
    p_dec4: ndarray = p_0 * np.ones(T, dtype=np.float32)
    for i in range(1, T): p_dec4[i] = w4[i] * p_dec4[i-1] + (1 - w4[i]) * p_inf
    
    fig, ax = plt.subplots()
    ax.axis('off')

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(t, p, label='p')
    ax1.plot(t, p_dec1, label='p_dec1')
    ax1.plot(t, p_dec2, label='p_dec2')
    ax1.plot(t, p_dec3, label='p_dec3')
    ax1.plot(t, p_dec4, label='p_dec4')
    ax1.set_xlabel('t'); ax1.set_ylabel('v')
    ax1.legend()

    fig.savefig('dp_single_version2/check/ppc.svg')

def main2():
    print(1111)

if __name__ == '__main__':
    default_config()
    main()