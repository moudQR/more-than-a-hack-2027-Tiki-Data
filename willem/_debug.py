import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'solution')
from config import PITCH
from features import extract_frame_features_metrica
from train import _frames_from_events
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd

g = Path('solution/external/metrica/Sample_Game_1')
X = extract_frame_features_metrica(g/'home.csv', g/'away.csv', PITCH)
y = _frames_from_events(pd.read_csv(g/'events.csv'), len(X), home_team='home')

keep = ~np.isnan(X[:, 1:]).all(axis=1)
X, y = X[keep], y[keep]
active = np.array([j for j in range(X.shape[1])
                   if len(np.unique(X[~np.isnan(X[:, j]), j])) >= 2])

clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0)
clf.fit(X[:, active], y)
pred = clf.predict(X[:, active])

print('Distribution on TRAINING DATA (in-sample, will be optimistic):')
import collections
print('true:', dict(collections.Counter(y)))
print('pred:', dict(collections.Counter(pred)))