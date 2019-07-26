from mlsquare.utils.correlations import concordance_correlation_coefficient
import numpy as np


def test_ccc():
    n_samples = 4
    y_true = np.arange(n_samples)
    y_pred = y_true
    c = concordance_correlation_coefficient(y_true,y_pred)
    np.testing.assert_allclose(c, 1)