import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from hns.eval.metrics import binary_metrics, recall_at_k


def test_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .1, .2, .3, .4, .5, .6, .7, .05])
    assert recall_at_k(y, s, 0.1) == 1 / 3  # top-1 is illicit
    assert recall_at_k(y, s, 1.0) == 1.0


def test_threshold_matches_sklearn_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(scale=0.8, size=400) > 1).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X[:300], y[:300])  # even count -> ties at 0.5
    m = binary_metrics(y[300:], rf.predict_proba(X[300:])[:, 1], ts=np.repeat([35, 36], 50))
    assert np.isclose(m["illicit_f1"], f1_score(y[300:], rf.predict(X[300:])))
    assert set(m["illicit_f1_per_ts"]) == {35, 36}
    assert sum(map(sum, m["confusion"])) == 100
