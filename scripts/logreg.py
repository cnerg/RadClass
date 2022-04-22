# For hyperopt (parameter optimization)
# ! pip install hyperopt
from hyperopt import STATUS_OK

# sklearn models
from sklearn import linear_model

# diagnostics
from sklearn.metrics import balanced_accuracy_score


def f_lr(params):
    # supervised logistic regression
    slr = linear_model.LogisticRegression(random_state=0, max_iter=params['max_iter'], tol=params['tol'], C=params['C'])#, multi_class='multinomial')
    slr.fit(trainx, trainy)
    slr_pred = slr.predict(testx)
    acc = balanced_accuracy_score(testy, slr_pred)

    return {'loss': 1-acc,
            'status': STATUS_OK,
            'model': slr,
            'params': params,
            'accuracy': acc}
