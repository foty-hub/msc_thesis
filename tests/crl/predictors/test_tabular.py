import numpy as np
from crl.predictors.tabular import (
    Predictor,
    NoPredictor,
    PredictorGlobal,
    PredictorSAConditioned,
)


def test_no_predictor_is_subclass():
    assert issubclass(NoPredictor, Predictor)


def test_predictor_global_is_subclass():
    assert issubclass(PredictorGlobal, Predictor)


def test_predictor_sa_conditioned_is_subclass():
    assert issubclass(PredictorSAConditioned, Predictor)


def test_global_predictor_conformal_coverage():
    """
    Tests that the GlobalPredictor achieves the desired coverage level.
    """
    alpha = 0.1
    n_calib = 1000
    n_test = 500
    n_states = 10

    # Note: min_count is set to n_calib to ensure the calibration set is full before providing non-trivial sets
    predictor = PredictorGlobal(alpha=alpha, n_calib=n_calib, min_count=n_calib)

    # --- Calibration Phase ---
    # We need to populate the calibration set.
    # The non-conformity score is 1 - p(s_prime | s, a), where p is the model's prediction.
    # We'll simulate a model's predictions.
    np.random.seed(42)

    # Generate calibration data
    calib_preds = np.random.rand(n_calib, n_states)
    calib_preds /= calib_preds.sum(axis=1, keepdims=True)
    calib_true_s_prime = np.random.randint(0, n_states, size=n_calib)

    for i in range(n_calib):
        # The observation tuple doesn't matter for the global predictor
        obs = (0, 0, 0.0, calib_true_s_prime[i])
        # The world model probability is the predicted probability of the true next state
        wm_prob = calib_preds[i, calib_true_s_prime[i]]
        predictor.observe(obs, wm_prob)

    # --- Testing Phase ---
    # Generate test data
    test_preds = np.random.rand(n_test, n_states)
    test_preds /= test_preds.sum(axis=1, keepdims=True)
    test_true_s_prime = np.random.randint(0, n_states, size=n_test)

    coverage_count = 0
    for i in range(n_test):
        # Get the prediction set
        # State and action don't matter for the global predictor
        prediction_set = predictor.conformalise(preds=test_preds[i], state=0, action=0)

        # Check if the true next state is in the prediction set
        if test_true_s_prime[i] in prediction_set:
            coverage_count += 1

    coverage = coverage_count / n_test

    # --- Assertion ---
    # The coverage should be approximately 1 - alpha.
    # We allow for some statistical fluctuation.
    # For n_test=500 and alpha=0.1, the expected number of misses is 50.
    # The standard deviation is sqrt(500 * 0.1 * 0.9) = sqrt(45) ~= 6.7
    # A 3-sigma interval for misses is 50 +/- 3*6.7 = 50 +/- 20.1 => [29.9, 70.1]
    # So coverage should be between (500-70.1)/500 and (500-29.9)/500
    # which is [0.8598, 0.9402]
    assert 0.85 < coverage < 0.95


# TODO: test that observe functions are working correctly - after an episode is the calibration set T-1 steps longer?
