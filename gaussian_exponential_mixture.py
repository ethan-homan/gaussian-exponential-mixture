import math
from copy import deepcopy
import numpy as np
from scipy import stats


class GaussianExponentialParameters:
    """Holds the parameters used in GaussianExponentialMixture.

    This class allows for access to parameters by name, pretty-printing,
    and comparison to other parameters to check for convergence.

    Args:
        beta (float): the scale parameter and mean for the exponential
            distribution this also corresponds to the mean, or the
            inverse of the rate of the exponential distribution.
        mu (float): the location parameter and mean for the gaussian
            distribution.
        sigma (float): the scale parameter and the standard deviation
            of the gaussian distribution.
        proportion (float): the proportion of the data that is likelier
            to be gaussian.
        exl_loc (float): the location of the start of the exponential distribution.
    """

    def __init__(self, beta=1.0, mu=0.0, sigma=100.0, proportion=0.5, exp_loc=0, **kwargs):
        self.beta: float = kwargs.get('beta', beta)
        self.mu: float = kwargs.get('mu', mu)
        self.sigma: float = kwargs.get('sigma', sigma)
        self.proportion: float = kwargs.get('proportion', proportion)
        self.exp_loc: float = kwargs.get('exp_loc', exp_loc)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'beta: {self.beta:.5f} | mu: {self.mu:.5f} | ' \
               f'sigma: {self.sigma:.5f} | exp_loc: {self.exp_loc:.5f} \
                | proportion: {self.proportion:.5f}'

    def as_list(self) -> list:
        """Gets the parameters as a list.

        Returns:
            beta, mu, sigma, and proportion as a list
        """
        return [self.beta, self.mu, self.sigma, self.proportion, self.exp_loc]

    def max_parameter_difference(self, other) -> float:
        """Get the largest difference in parameters to another GaussianExponentialParameters.

        Compares this object to another GaussianExponentialParameters object parameter by
        parameter and returns the absolute value of the largest difference.

        Args:
            other (GaussianExponentialParameters): the parameters to compare to. This operation
                is symmetric.

        Returns:
            The largest pairwise difference in the parameter list.
        """
        return max([abs(i[0] - i[1]) for i in zip(self.as_list(), other.as_list())])


class GaussianExponentialMixture:

    """Fits a mixture of a Gaussian and Exponential distribution to data in a Numpy array

    This implementation uses Expectation Maximization -- referred to as EM in these docs --
    to iteratively converge on solutions for four unknown parameters:

        - mu: the mean of the Gaussian/Normal distribution
        - sigma: the standard deviation of the Gaussian/Normal distribution
        - beta: the mean of the Exponential distribution
        - proportion: the proportion of the data that is gaussian

    TODO: Link to Appendix with derivations of update conditions.

    Args:
        data (np.numarray): single dimensional array of data to fit distributions to
        exp_loc (float): location of the exponential distribution
        max_iterations (int): terminate after this number of EM steps
        convergence_tolerance (float): terminate if no parameter moves by more than this value
        distribution_fix (bool): support use case where gaussian mu and exponential offset are locked
    """

    def __init__(self,
                 data: np.numarray,
                 max_iterations=100,
                 convergence_tolerance=0.001,
                 distribution_fix=False,
                 **kwargs):

        self.convergence_tolerance: float = convergence_tolerance
        self.data: np.numarray = data
        self.parameters = GaussianExponentialParameters( **kwargs)
        self.parameters_updated = GaussianExponentialParameters( **kwargs)
        self.max_iterations: int = max_iterations
        self.distribution_fix: bool = distribution_fix
        self.expon = stats.expon(loc=self.parameters.exp_loc, scale=self.parameters.beta)
        self.norm = stats.norm(loc=self.parameters.mu, scale=self.parameters.sigma)

    def _apply_and_sum(self, func: callable) -> float:
        """Applies a function to the data and returns the sum of the array.

        Args:
            func (callable): a callable with the signature func(val: float) -> float.

        Returns:
            The sum of the data vector after applying func.
        """
        return np.sum(np.vectorize(func)(self.data))

    def _expectation_is_gaussian(self, val: float) -> float:
        """Computes (prob_gaussian)/(prob_gaussian + prob_exponential) for the value passed
           with some protection against underflow.
        """
        gaussian_density = self.norm.logpdf(val)
        exponential_density = self.expon.logpdf(val)
        log_prob_gaussian = gaussian_density + np.log(self.parameters.proportion)
        log_prob_exponential = exponential_density + np.log(1 - self.parameters.proportion)
        expectation_is_gaussian = np.exp(
                log_prob_gaussian - np.logaddexp(log_prob_gaussian, log_prob_exponential)
        )
        if expectation_is_gaussian == np.nan:
            return 0
        else:
            return expectation_is_gaussian

    def _update_beta(self) -> None:
        """Updates the beta parameter (mean/scale) of the exponential distribution.
        """
        self.parameters_updated.beta = \
            self._apply_and_sum(lambda x: (1 - self._expectation_is_gaussian(x)) * (x-self.parameters_updated.exp_loc)) / \
            self._apply_and_sum(lambda x: (1 - self._expectation_is_gaussian(x)))

    def _update_mu(self) -> None:
        """Updates the mu parameter (mean/location) of the gaussian distribution.
        """
        self.parameters_updated.mu = \
            self._apply_and_sum(lambda x: self._expectation_is_gaussian(x) * x) / \
            self._apply_and_sum(lambda x: self._expectation_is_gaussian(x))

    def _update_exp_loc(self) -> None:
        """Updates the location parameter of the exponential distribution.

         Note:
            Assumes this parameter is fixed unless it track the Gausian Mu.  There might be a update
            equation for the normal case that could be added in future
        """
        if self.distribution_fix is True:
           self.parameters_updated.exp_loc = self.parameters_updated.mu #+ (2*self.parameters_updated.sigma)
           
    def _update_sigma(self) -> None:
        """Updates the sigma parameter (standard deviation/scale) of the gaussian distribution.

        Note:
            Updating the standard deviation of the normal distribution requires the updated
            mean for this iteration to be in updated_parameters for behavior to be defined.
        """
        sigma_squared = \
            self._apply_and_sum(lambda x: (self._expectation_is_gaussian(x)) * (x - self.parameters_updated.mu) ** 2) / \
            self._apply_and_sum(lambda x: (self._expectation_is_gaussian(x)))
        self.parameters_updated.sigma = math.sqrt(sigma_squared)

    def _update_proportion(self) -> None:
        """Updates the proportion of the data that is likelier gaussian.
        """
        gaussian_total = self._apply_and_sum(lambda x: np.nan_to_num(self.norm.logpdf(x)) >
                                                       np.nan_to_num(self.expon.logpdf(x)))
        self.parameters_updated.proportion = gaussian_total / len(self.data)

    def _sync_parameters(self) -> None:
        """Copies parameters_updated into parameters.

        This prepares the state of GaussianExponentialMixture for another iteration
        of the EM algorithm with the parameters updated from the previous iteration.
        """
        self.parameters = deepcopy(self.parameters_updated)

    def _update_pdfs(self) -> None:
        """Updates PDFs of normal and exponential with new parameters.

        Since the parameters are stored separately from the PDFs for now, updates
        need to be applied on each iteration.
        """
        self.norm = stats.norm(loc=self.parameters_updated.mu, scale=self.parameters_updated.sigma)
    #    if self.distribution_fix is False:
    #        self.expon = stats.expon(loc=self._exp_loc, scale=self.parameters_updated.beta)
    #    else:
        self.expon = stats.expon(loc=self.parameters_updated.exp_loc, scale=self.parameters_updated.beta)


    def _check_parameter_differences(self) -> float:
        """Compares the newly updated parameters to the previous iteration.

        Returns:
            This returns the largest pairwise difference between parameter values for
            use in determining the convergence of EM.
        """
        return self.parameters.max_parameter_difference(self.parameters_updated)

    def em_step(self) -> None:
        """Performs one EM step on the data and stores the result in updated_parameters.

        Note:
            While This method can be used safely independently, it is advisable to use `self.fit`
            in almost all cases outside of debugging since it handles a iteration and
            tracks convergence.
        """
        self._sync_parameters()
        self._update_beta()
        self._update_mu()
        self._update_exp_loc()
        self._update_sigma()
        self._update_pdfs()
        self._update_proportion()

    def fit(self) -> None:
        """Performs EM steps until convergence criteria are satisfied.

        Note:
            If your data is large or your convergence criteria is strict this may take
            a long time.

            To debug, consider running `em_step` directly and monitoring parameter movement
            and iteration time.
        """
        self.em_step()
        iters = 1
        while iters < self.max_iterations and self._check_parameter_differences() > self.convergence_tolerance:
            self.em_step()
            iters += 1
        self._sync_parameters()

    def logpdf(self, val):
        """Evaluates the density of the logpdf of the GaussianExponentialMixture.
        """
        weighted_log_gaussian_density = np.log(self.parameters.proportion) + self.norm.logpdf(val)
        weighted_log_exponential_density = np.log((1 - self.parameters.proportion)) + self.expon.logpdf(val)
        log_density = np.logaddexp(weighted_log_gaussian_density, weighted_log_exponential_density)
        return log_density

    def pdf(self, val):
        """Evaluates the density of the pdf of the GaussianExponentialMixture.
        """
        return np.exp(self.logpdf(val))
