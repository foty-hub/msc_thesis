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


# TODO: tests of conformalise - check correct conformal sets on a known small dataset
# TODO: test that observe functions are working correctly - after an episode is the calibration set T-1 steps longer?
