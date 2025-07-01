from crl.predictors.tabular import (
    Predictor,
    NoPredictor,
    PredictorGlobal,
    PredictorSAConditioned,
)

assert issubclass(NoPredictor, Predictor)
assert issubclass(PredictorGlobal, Predictor)
assert issubclass(PredictorSAConditioned, Predictor)
