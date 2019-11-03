from gaussian_exponential_mixture import GaussianExponentialMixture


class TestGaussianExponentialMixture:

    @classmethod
    def setup_class(cls):
        cls.data = [1, 1, 1, 5, 6, 6, 6, 7]
        cls.gme = GaussianExponentialMixture(cls.data)

    def test_apply_and_sum(self):
        add_one_result = self.gme._apply_and_sum(lambda x: x + 1)
        add_one_expected = sum(self.data) + len(self.data)
        assert(add_one_result == add_one_expected)

    def test_fit(self):
        self.gme.fit()
        assert(int(self.gme.parameters.mu) == 6)
