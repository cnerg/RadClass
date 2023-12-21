# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# sklearn models
from sklearn import linear_model
# diagnostics
from sklearn.metrics import balanced_accuracy_score
from scripts.utils import run_hyperopt
import joblib


class LogReg:
    '''
    Methods for deploying sklearn's logistic regression
    implementation with hyperparameter optimization.
    Data agnostic (i.e. user supplied data inputs).
    TODO: Currently only supports binary classification.
        Add multinomial functions and unit tests.
        Add functionality for regression(?)
    Inputs:
    kwargs: logistic regression input functions.
        keys random_state, max_iter, tol, and C supported.
    random_state: int/float for reproducible intiailization.
    '''

    # only binary so far
    def __init__(self, **kwargs):
        # supported keys = ['max_iter', 'tol', 'C', 'random_state']
        # defaults to a fixed value for reproducibility
        self.random_state = kwargs.pop('random_state', 0)
        # parameters for logistic regression model:
        # defaults to sklearn.linear_model.LogisticRegression default vals
        self.max_iter = kwargs.pop('max_iter', 100)
        self.tol = kwargs.pop('tol', 0.0001)
        self.C = kwargs.pop('C', 1.0)
        self.model = linear_model.LogisticRegression(
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        C=self.C
                    )

    def fresh_start(self, params, data_dict):
        '''
        Required method for hyperopt optimization.
        Trains and tests a fresh logistic regression model
        with given input parameters.
        This method does not overwrite self.model (self.optimize() does).
        Inputs:
        params: dictionary of logistic regression input functions.
            keys max_iter, tol, and C supported.
        data_dict: compact data representation with the four requisite
            data structures used for training and testing a model.
            keys trainx, trainy, testx, and testy required.
        '''

        # unpack data
        trainx = data_dict['trainx']
        trainy = data_dict['trainy']
        testx = data_dict['testx']
        testy = data_dict['testy']

        # supervised logistic regression
        clf = LogReg(params=params, random_state=self.random_state)
        # train and test model
        clf.train(trainx, trainy)
        # uses balanced_accuracy accounts for class imbalanced data
        clf_pred, acc = clf.predict(testx, testy)

        # loss function minimizes misclassification
        return {'loss': 1-acc,
                'status': STATUS_OK,
                'model': clf.model,
                'params': params,
                'accuracy': acc}

    def optimize(self, space, data_dict, max_evals=50, verbose=True):
        '''
        Wrapper method for using hyperopt (see utils.run_hyperopt
        for more details). After hyperparameter optimization, results
        are stored, the best model -overwrites- self.model, and the
        best params -overwrite- self.params.
        Inputs:
        space: a hyperopt compliant dictionary with defined optimization
            spaces. For example:
                # quniform returns float, some parameters require int;
                # use this to force int
                space = {'max_iter': scope.int(hp.quniform('max_iter',
                                                           10,
                                                           10000,
                                                           10)),
                        'tol'      : hp.loguniform('tol', 1e-5, 1e-1),
                        'C'        : hp.uniform('C', 0.001,1000.0)
                        }
            See hyperopt docs for more information.
        data_dict: compact data representation with the four requisite
            data structures used for training and testing a model.
            keys trainx, trainy, testx, testy required.
        max_evals: the number of epochs for hyperparameter optimization.
            Each iteration is one set of hyperparameters trained
            and tested on a fresh model. Convergence for simpler
            models like logistic regression typically happens well
            before 50 epochs, but can increase as more complex models,
            more hyperparameters, and a larger hyperparameter space is tested.
        verbose: boolean. If true, print results of hyperopt.
            If false, print only the progress bar for optimization.
        '''

        best, worst = run_hyperopt(space=space,
                                   model=self.fresh_start,
                                   data_dict=data_dict,
                                   max_evals=max_evals,
                                   verbose=verbose)

        # save the results of hyperparameter optimization
        self.best = best
        self.model = best['model']
        self.params = best['params']
        self.worst = worst

    def train(self, trainx, trainy):
        '''
        Wrapper method for sklearn's logisitic regression training method.
        Inputs:
        trainx: nxm feature vector/matrix for training model.
        trainy: nxk class label vector/matrix for training model.
        '''

        # supervised logistic regression
        self.model.fit(trainx, trainy)

    def predict(self, testx, testy=None):
        '''
        Wrapper method for sklearn's logistic regression predict method.
        Inputs:
        testx: nxm feature vector/matrix for testing model.
        testy: nxk class label vector/matrix for training model.
            optional: if included, the predicted classes -and-
            the resulting classification accuracy will be returned.
        '''

        pred = self.model.predict(testx)

        acc = None
        if testy is not None:
            # uses balanced_accuracy_score to account for class imbalance
            acc = balanced_accuracy_score(testy, pred)

        return pred, acc

    def save(self, filename):
        '''
        Save class instance to file using joblib.
        Inputs:
        filename: string filename to save object to file under.
            The file must be saved with extension .joblib.
            Added to filename if not included as input.
        '''

        if filename[-7:] != '.joblib':
            filename += '.joblib'
        joblib.dump(self, filename)
