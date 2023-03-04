import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn import cluster
from scipy.stats import poisson, norm

class BaseHmm:
    def __init__(self, n_state, n_iter=10):
        self.n_state = n_state
        self.n_iter = n_iter

    def init(self, data):
        init_p = 1. / self.n_state
        self.startprob = np.random.dirichlet(np.full(self.n_state, init_p))  # (n_state, )
        self.transmat = np.random.dirichlet(np.full(self.n_state, init_p),
                                            size=self.n_state)  # (n_state, n_state)    sum(row) = 1  row=from, col=to

        _, self.n_feature = data.shape

    def compute_log_likelihood(self, data):
        return

    def forward_pass(self, data, emiss_prob_log):
        alpha_log = np.log(self.startprob)[np.newaxis, ...] + self.compute_log_likelihood(data)

        for t in range(1, len(data)):
            for j in range(self.n_state):
                ep_log = emiss_prob_log[t, j]
                alpha_log[t, j] = np.log(np.sum(np.exp(alpha_log[t-1] + np.log(self.transmat[:, j]) + ep_log)))
        return alpha_log

    def backward_pass(self, data, emiss_prob_log):
        beta_log = np.zeros((len(data), self.n_state))  # log space

        for t in range(len(data)-2, -1, -1):
            for i in range(self.n_state):
                ep_log = emiss_prob_log[t+1]    # (n_state, )
                beta_log[t, i] = np.log(np.sum(np.exp(np.log(self.transmat[i]) + ep_log + beta_log[t+1])))

        return beta_log

    def calculate_gamma(self, alpha_log, beta_log):
        numerator = alpha_log + beta_log
        divider = np.log(np.sum(np.exp(numerator), 1))

        gamma_log = numerator - divider[..., np.newaxis]
        return gamma_log

    def calculate_xi(self, alpha_log, beta_log, emiss_prob_log):
        xi_log = np.empty((len(alpha_log)-1, self.n_state, self.n_state))

        divider = np.log(np.sum(np.exp(alpha_log + beta_log), 1))

        for t in range(len(alpha_log)-1):
            for i in range(self.n_state):
                for j in range(self.n_state):
                    xi_log[t, i, j] = (alpha_log[t, i] + np.log(self.transmat[i, j]) + emiss_prob_log[t+1, j] + beta_log[t+1, j]) - divider[t]
        return xi_log

    def e_step(self, data, emiss_prob_log):
        # this part is forward-backward algorithm
        alpha_log = self.forward_pass(data, emiss_prob_log)
        beta_log = self.backward_pass(data, emiss_prob_log)

        gamma_log = self.calculate_gamma(alpha_log, beta_log)
        xi_log = self.calculate_xi(alpha_log, beta_log, emiss_prob_log)

        return gamma_log, xi_log

    def m_step(self, data, gamma_log, xi_log, emiss_prob_log):
        # mostly get from https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py#L1031
        # update transition probability
        numerator = np.sum(np.exp(xi_log), 0)
        divider = np.sum(np.exp(xi_log), (2, 0))[..., np.newaxis]
        self.transmat = numerator / divider

        # update starting probability
        numerator = np.exp(gamma_log)[0]
        divider = np.sum(numerator)
        self.startprob = numerator / divider

    def fit(self, data):
        self.init(data)

        for iter in range(self.n_iter):
            self.fit_log(data)

    def fit_log(self, data):    # Baum-Welch algorithm
        emiss_prob_log = self.compute_log_likelihood(data)  # (len(data)=time, n_state)

        gamma_log, xi_log = self.e_step(data, emiss_prob_log)
        self.m_step(data, gamma_log, xi_log, emiss_prob_log)

    def generate_from_state(self, state):
        return

    def sample(self, n_sample):
        samples = []
        state_seq = []
        transmat_cdf = np.cumsum(self.transmat, 1)
        startprob_cdf = np.cumsum(self.startprob)

        cur_state = (startprob_cdf > np.random.rand()).argmax()
        samples.append(self.generate_from_state(cur_state))
        state_seq.append(cur_state)

        for t in range(n_sample-1):
            cur_state = (transmat_cdf[cur_state] > np.random.rand()).argmax()
            samples.append(self.generate_from_state(cur_state))
            state_seq.append(cur_state)

        return np.array(samples), np.array(state_seq)

    def decode(self):
        return

class PoissonHmm(BaseHmm):   # which means emission probability follows Poisson distribution
    def __init__(self, n_state, n_iter=10):
        super(PoissonHmm, self).__init__(n_state, n_iter)

    def init(self, data):
        super().init(data)

        mean = np.mean(data)
        var = np.var(data)

        self.lambdas = np.random.gamma(shape=mean ** 2 / var, scale=var / mean, size=(self.n_state, self.n_feature))    # (n_state, )

    def compute_log_likelihood(self, data):
        return np.array([np.sum(poisson.logpmf(data, lam), axis=1) for lam in self.lambdas]).T    # (len(data)=time, n_state)

    def m_step(self, data, gamma_log, xi_log, emiss_prob_log):
        super().m_step(data, gamma_log, xi_log, emiss_prob_log)

        n = np.sum(np.exp(gamma_log))

        post = np.sum(np.exp(gamma_log), 0)
        obs = np.dot(np.exp(gamma_log).T, data)
        y_bar = obs / post[:, np.newaxis]
        self.lambdas = (n * y_bar) / n

    def decode(self, data):
        # Viterbi algorithm
        v = np.zeros((len(data), self.n_state))
        emiss_prob_log = self.compute_log_likelihood(data)
        v[0] = np.log(self.startprob) + emiss_prob_log[0]

        for t in range(1, len(data)):
            for j in range(self.n_state):
                v[t, j] = np.max(v[t-1] + np.log(self.transmat)[:, j] + emiss_prob_log[t, j])

        idx_seq = np.argmax(v, 1)
        state_seq = self.lambdas[idx_seq]
        return state_seq

    def generate_from_state(self, state):
        return np.random.poisson(self.lambdas[state])

class GaussianHmm(BaseHmm):
    def __init__(self, n_state, n_iter=10):
        super(GaussianHmm, self).__init__(n_state, n_iter)

    def init(self, data):
        super().init(data)

        kmeans = cluster.KMeans(n_clusters=self.n_state)
        kmeans.fit(data)
        self.mean = np.squeeze(kmeans.cluster_centers_)
        cv = np.cov(data.T) + np.eye(data.shape[1])
        self.cov = np.sqrt(np.squeeze(np.tile(np.diag(cv), (self.n_state, 1))))

    def compute_log_likelihood(self, data):
        return np.array([np.sum(norm.logpdf(data, loc=m, scale=v), axis=1) for m, v in zip(self.mean, self.cov)]).T

    def generate_from_state(self, state):
        return np.random.normal(loc=self.mean[state], scale=self.cov[state])

    def m_step(self, data, gamma_log, xi_log, emiss_prob_log):
        super().m_step(data, gamma_log, xi_log, emiss_prob_log)

        self.mean = np.squeeze(np.dot(np.exp(gamma_log).T, data)) / np.sum(np.exp(gamma_log), 0)
        self.cov = np.sqrt(np.sum(np.exp(gamma_log) * np.square(data - self.mean), 0) / np.sum(np.exp(gamma_log), 0))

    def decode(self, data):
        # Viterbi algorithm
        v = np.zeros((len(data), self.n_state))
        emiss_prob_log = self.compute_log_likelihood(data)
        v[0] = np.log(self.startprob) + emiss_prob_log[0]

        for t in range(1, len(data)):
            for j in range(self.n_state):
                v[t, j] = np.max(v[t-1] + np.log(self.transmat)[:, j] + emiss_prob_log[t, j])

        idx_seq = np.argmax(v, 1)
        state_seq = self.mean[idx_seq]

        return state_seq[..., np.newaxis]