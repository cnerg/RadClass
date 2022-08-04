import numpy as np
# For hyperopt (parameter optimization)
from scripts.optimize import STATUS_OK
# sklearn models
from sklearn.semi_supervised import LabelPropagation
# diagnostics
from sklearn.metrics import balanced_accuracy_score

lp_trainx = np.append(trainx, U[:,1:], axis=0)
lp_trainy = np.append(trainy, U[:,0], axis=0)


def f_lp(params):
    lp = LabelPropagation(kernel='knn', gamma=params['gamma'], n_neighbors=params['n_neighbors'], max_iter=params['max_iter'], tol=params['tol'], n_jobs=-1)
    lp.fit(lp_trainx, lp_trainy)
    acc = balanced_accuracy_score(testy, lp.predict(testx))

    return {'loss': 1-acc,
            'status': STATUS_OK,
            'model': lp,
            'params': params,
            'accuracy': acc}