"""Pre-solve baseline: raw question only; no trajectory or benchmark/ID inputs."""
import re
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

ACTIONS = ('stop', 'continue_512', 'continue_2048')


class QuestionStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix([[np.log1p(len(t)), np.log1p(len(t.split())),
            np.log1p(len(re.findall(r'\d+', t))), np.log1p(t.count('\\')),
            np.log1p(sum(t.count(x) for x in '=+-*/^')), np.log1p(t.count('\n')),
            np.log1p(t.count('?')), np.log1p(t.count('$'))] for t in X], dtype=np.float64)


def features(mode):
    parts = [('stats', Pipeline([('stats', QuestionStats()), ('scale', MaxAbsScaler())]))]
    if mode in ('word', 'both'):
        parts.append(('word', TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                            max_features=12000, sublinear_tf=True)))
    if mode in ('char', 'both'):
        parts.append(('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                            min_df=2, max_features=18000, sublinear_tf=True)))
    return FeatureUnion(parts)


def predict(bundle, questions):
    if any(not isinstance(q, str) for q in questions):
        raise TypeError('Predictor accepts question strings only')
    x = bundle['features'].transform(questions)
    if bundle.get('svd') is not None:
        x = bundle['svd'].transform(x)
    p = np.stack([np.clip(m.predict(x), 0, 1) for m in bundle['success']], axis=1)
    c = np.stack([np.maximum(m.predict(x), 0) for m in bundle['cost']], axis=1)
    return {'p_success': p, 'predicted_total_generated_tokens': c}
