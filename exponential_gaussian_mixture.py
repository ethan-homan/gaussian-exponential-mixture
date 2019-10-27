import math
from copy import deepcopy
import numpy
from scipy import stats


class Parameters(object):
    def __init__(self, beta=1.0, mu=0.0, sigma=100.0, proportion=0.5, **kwargs):
        self.beta: float = kwargs.get('beta', beta)
        self.mu: float = kwargs.get('mu', mu)
        self.sigma: float = kwargs.get('sigma', sigma)
        self.proportion: float = kwargs.get('proportion', proportion)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'beta: {self.beta:.5f} | mu: {self.mu:.5f} | ' \
               f'sigma: {self.sigma:.5f} | proportion: {self.proportion:.5f}'

    def as_list(self) -> list:
        return [self.beta, self.mu, self.sigma, self.proportion]

    def max_parameter_difference(self, other) -> float:
        return max([abs(i[0] - i[1]) for i in zip(self.as_list(), other.as_list())])


class GaussianExponentialMixture(object):

    def __init__(self,
                 data: numpy.numarray,
                 exp_loc=0.0,
                 max_iterations=100,
                 convergence_tolerance=0.001,
                 **kwargs):

        self.convergence_tolerance = convergence_tolerance
        self.data: numpy.numarray = data
        self._exp_loc: float = exp_loc
        self.parameters: Parameters = Parameters(**kwargs)
        self.parameters_updated: Parameters = Parameters(**kwargs)
        self.expon = stats.expon(loc=self._exp_loc, scale=self.parameters.beta)
        self.norm = stats.norm(loc=self.parameters.mu, scale=self.parameters.sigma)
        self.max_iterations: int = max_iterations

    def _apply_and_sum_over_data(self, func) -> float:
        return sum(numpy.vectorize(func)(self.data))

    def _evaluate_pi(self, val: float) -> float:
        gaussian_density = self.norm.pdf(val)
        exponential_density = self.expon.pdf(val)
        if exponential_density == numpy.nan:
            return 1
        elif gaussian_density == numpy.nan:
            return 0
        elif self.parameters.proportion == 0:
            return 0
        probability_gaussian = gaussian_density * self.parameters.proportion
        probability_exponential = exponential_density * (1 - self.parameters.proportion)
        return probability_gaussian / (probability_gaussian + probability_exponential)

    def _update_beta(self) -> None:
        self.parameters_updated.beta = \
            self._apply_and_sum_over_data(lambda x: (1 - self._evaluate_pi(x)) * x) / \
            self._apply_and_sum_over_data(lambda x: (1 - self._evaluate_pi(x)))

    def _update_mu(self) -> None:
        self.parameters_updated.mu = \
            self._apply_and_sum_over_data(lambda x: self._evaluate_pi(x) * x) / \
            self._apply_and_sum_over_data(lambda x: self._evaluate_pi(x))

    def _update_sigma(self) -> None:
        sigma_squared = \
            self._apply_and_sum_over_data(lambda x: (self._evaluate_pi(x)) * (x - self.parameters_updated.mu) ** 2) / \
            self._apply_and_sum_over_data(lambda x: (self._evaluate_pi(x)))
        self.parameters_updated.sigma = math.sqrt(sigma_squared)

    def _update_proportion(self) -> None:
        gaussian_total = self._apply_and_sum_over_data(lambda x: numpy.nan_to_num(self.norm.logpdf(x)) >
                                                                 numpy.nan_to_num(self.expon.logpdf(x)))
        self.parameters_updated.proportion = gaussian_total / len(self.data)

    def _sync_parameters(self) -> None:
        self.parameters = deepcopy(self.parameters_updated)

    def _update_pdfs(self) -> None:
        self.norm = stats.norm(loc=self.parameters_updated.mu, scale=self.parameters_updated.sigma)
        self.expon = stats.expon(loc=self._exp_loc, scale=self.parameters_updated.beta)

    def _check_parameter_differences(self) -> float:
        return self.parameters.max_parameter_difference(self.parameters_updated)

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
