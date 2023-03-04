import matplotlib.pyplot as plt
import numpy as np
from hmm import PoissonHmm, GaussianHmm

def main():
    data = np.array([
        13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18,
        25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26,
        13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26,
        28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20,
        15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
        18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
        15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11])

    models = [
        PoissonHmm(n_state=5),
        GaussianHmm(n_state=5)
    ]

    for model in models:

        model.fit(data[..., np.newaxis])

        state_seq = model.decode(data[..., np.newaxis])

        fig = plt.figure()
        plt.plot(np.arange(len(data)), data, label='original')
        plt.plot(np.arange(len(data)), state_seq[:, 0], label='states')
        plt.legend()
        if isinstance(model, PoissonHmm):
            plt.title('Poisson (decode)')
        elif isinstance(model, GaussianHmm):
            plt.title('Gaussian (decode)')
        fig.show()

        samples, state_seq = model.sample(100)

        fig = plt.figure()
        if isinstance(model, PoissonHmm):
            plt.plot(np.arange(len(state_seq)), model.lambdas[state_seq], label='states')
        elif isinstance(model, GaussianHmm):
            plt.plot(np.arange(len(state_seq)), model.mean[state_seq], label='states')
        plt.plot(np.arange(len(samples)), samples, label='samples')
        plt.legend()
        if isinstance(model, PoissonHmm):
            plt.title('Poisson (sample)')
        elif isinstance(model, GaussianHmm):
            plt.title('Gaussian (sample)')
        fig.show()
    plt.show()

if __name__ == '__main__':
    main()