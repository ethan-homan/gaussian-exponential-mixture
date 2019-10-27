from scipy import stats
import math
import numpy


class Parameters(object):
    def __init__(self, beta=1.0, mu=0.0, sigma=100.0, proportion=0.5):
        self.beta: float = beta
        self.mu: float = mu
        self.sigma: float = sigma
        self.proportion: float = proportion

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'beta: {self.beta:.5f} | mu: {self.mu:.5f} | sigma: {self.sigma:.5f} | proportion: {self.proportion:.5f}'

    def as_list(self):
        return [self.beta, self.mu, self.sigma, self.proportion]


class Mixture(object):
    pass


class GaussianExponentialMixture(Mixture):

    def __init__(self, data: numpy.numarray, max_iterations=100, convergence_tolerance=0.001):

        self.convergence_tolerance = convergence_tolerance
        self.data: numpy.numarray = data
        self.parameters: Parameters = Parameters()
        self.parameters_updated: Parameters = Parameters()
        self.expon = stats.expon(scale=self.parameters.beta)
        self.norm = stats.norm(loc=self.parameters.mu, scale=self.parameters.sigma)
        self.max_iterations = max_iterations

    def _apply_and_sum_over_data(self, f) -> float:
        return sum(numpy.vectorize(f)(self.data))

    def _evaluate_pi(self, x: float) -> float:
        g = self.norm.pdf(x)
        e = self.expon.pdf(x)
        if x < 0.0:
            return 1
        elif e == numpy.nan:
            return 1
        elif g == numpy.nan:
            return 0
        elif self.parameters.proportion == 0:
            return 0
        else:
            probability_gaussian = g * self.parameters.proportion
            probability_exponential = e * (1 - self.parameters.proportion)
            return probability_gaussian / (probability_gaussian + probability_exponential)

    def _update_beta(self) -> None:
        beta = self._apply_and_sum_over_data(lambda x: (1 - self._evaluate_pi(x)) * x) / \
               self._apply_and_sum_over_data(lambda x: (1 - self._evaluate_pi(x)))
        self.parameters_updated.beta = beta

    def _update_mu(self) -> None:
        self.parameters_updated.mu = self._apply_and_sum_over_data(lambda x: self._evaluate_pi(x) * x) / \
                  self._apply_and_sum_over_data(lambda x: self._evaluate_pi(x))

    def _update_sigma(self) -> None:
        sigma_squared = self._apply_and_sum_over_data(lambda x: (self._evaluate_pi(x)) * (x - self.parameters_updated.mu) ** 2) / \
                self._apply_and_sum_over_data(lambda x: (self._evaluate_pi(x)))
        self.parameters_updated.sigma = math.sqrt(sigma_squared)

    def _update_proportion(self) -> None:
        total = len(self.data)
        gaussian_total = self._apply_and_sum_over_data(lambda x: numpy.nan_to_num(self.norm.logpdf(x)) > numpy.nan_to_num(self.expon.logpdf(x)))
        self.parameters_updated.proportion = gaussian_total / float(total)

    def _sync_parameters(self) -> None:
        self.parameters.beta, self.parameters.mu, self.parameters.sigma, self.parameters.proportion = \
            self.parameters_updated.beta, self.parameters_updated.mu, self.parameters_updated.sigma, self.parameters_updated.proportion

    def _update_pdfs(self) -> None:
        self.norm = stats.norm(loc=self.parameters_updated.mu, scale=self.parameters_updated.sigma)
        self.expon = stats.expon(scale=self.parameters_updated.beta)

    def _check_parameter_differences(self) -> float:
        largest_parameter_change = max([abs(i[0] - i[1]) for i in zip(self.parameters.as_list(), self.parameters_updated.as_list())])
        return largest_parameter_change

    def _em_step(self) -> None:
        self._sync_parameters()
        self._update_beta()
        self._update_mu()
        self._update_sigma()
        self._update_pdfs()
        self._update_proportion()

    def fit(self) -> None:
        self._em_step()
        iters = 1
        while iters < self.max_iterations and self._check_parameter_differences() > self.convergence_tolerance:
            self._em_step()
            iters += 1
