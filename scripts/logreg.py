# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# sklearn models
from sklearn import linear_model
# diagnostics
from sklearn.metrics import balanced_accuracy_score
from scripts.hyperopt import run_hyperopt

class LogisticRegression:
    # only binary so far
    def __init__(self, params=None):
        # dictionary of parameters for logistic regression model
        self.params = params
        if self.params is None:
            self.model = linear_model.LogisticRegression()
        else:
            self.model = linear_model.LogisticRegression(random_state=0, max_iter=params['max_iter'], tol=params['tol'], C=params['C'])

    def fresh_start(self, params, data_dict):
        # unpack data
        trainx = data_dict['trainx']
        trainy = data_dict['trainy']
        testx = data_dict['testx']
        testy = data_dict['testy']

        # supervised logistic regression
        clr = linear_model.LogisticRegression(random_state=0, max_iter=params['max_iter'], tol=params['tol'], C=params['C'])
        clr.fit(trainx, trainy)
        clr_pred = clr.predict(testx)
        # could alternatively use pure accuracy for a more traditional hyperopt
        acc = balanced_accuracy_score(testy, clr_pred)

        return {'loss': 1-acc,
                'status': STATUS_OK,
                'model': clr,
                'params': params,
                'accuracy': acc}

    def optimize(self, space, max_evals=50, verbose=True):
        best, worst = run_hyperopt(space, self.fresh_start, max_evals, verbose)

        self.best = best
        self.model = best['model']
        self.params = best['params']
        self.worst = worst

    def train(self, trainx, trainy):
        # supervised logistic regression
        self.model.fit(trainx, trainy)

    def test(self, testx, testy=None):
        pred = self.model.predict(testx)

        acc = 0.
        if testy is not None:
            acc = balanced_accuracy_score(testy, pred)
        
        return pred, acc
