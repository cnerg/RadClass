import numpy as np
# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# sklearn models
from sklearn import semi_supervised
# diagnostics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from scripts.utils import run_hyperopt
import joblib


class LabelProp:
    '''
    Methods for deploying sklearn's Label Propagation
    implementation with hyperparameter optimization.
    Data agnostic (i.e. user supplied data inputs).
    NOTE: Since LabelProp is guaranteed to converge given
        enough iterations, there is no random_state defined.
    TODO: Currently only supports binary classification.
        Add multinomial functions and unit tests.
        Add functionality for regression(?)
    Inputs:
    params: dictionary of logistic regression input functions.
        keys gamma, n_neighbors, max_iter, and tol supported.
    alpha: float; weight for encouraging high recall
    beta: float; weight for encouraging high precision
    NOTE: if alpha=beta=0, default to favoring balanced accuracy.
    random_state: int/float for reproducible intiailization.
    '''

    # only binary so far
    def __init__(self, params=None, alpha=0, beta=0, random_state=0):
        # defaults to a fixed value for reproducibility
        self.random_state = random_state
        # dictionary of parameters for Label Propagation model
        self.alpha, self.beta = alpha, beta
        self.params = params
        if self.params is None:
            # defaults:
            # knn kernel, although an rbf is equally valid
            # TODO: allow rbf kernels
            # n_jobs, use parallelization if available.
            self.model = semi_supervised.LabelPropagation(
                            kernel='knn',
                            n_jobs=-1
                        )
        else:
            self.model = semi_supervised.LabelPropagation(
                            kernel='knn',
                            gamma=params['gamma'],
                            n_neighbors=params['n_neighbors'],
                            max_iter=params['max_iter'],
                            tol=params['tol'],
                            n_jobs=-1
                        )

    def fresh_start(self, params, data_dict):
        '''
        Required method for hyperopt optimization.
        Trains and tests a fresh Label Propagation model
        with given input parameters.
        This method does not overwrite self.model (self.optimize() does).
        Inputs:
        params: dictionary of logistic regression input functions.
            keys max_iter, tol, and C supported.
        data_dict: compact data representation with the five requisite
            data structures used for training and testing an SSML model.
            keys trainx, trainy, testx, testy, and Ux required.
            NOTE: Uy is not needed since labels for unlabeled data
            instances is not used.
        '''

        # unpack data
        trainx = data_dict['trainx']
        trainy = data_dict['trainy']
        testx = data_dict['testx']
        testy = data_dict['testy']
        Ux = data_dict['Ux']

        clf = LabelProp(params, random_state=self.random_state)
        # training and testing
        clf.train(trainx, trainy, Ux)
        # uses balanced_accuracy accounts for class imbalanced data
        pred, acc = clf.predict(testx, testy)
        rec = recall_score(testy, pred)
        prec = precision_score(testy, pred)

        # loss function minimizes misclassification
        # by maximizing metrics
        return {'score': acc+(self.alpha*rec)+(self.beta*prec),
                'loss': (1-acc) + self.alpha*(1-rec)+self.beta*(1-prec),
                'model': clf,
                'params': params,
                'accuracy': acc,
                'precision': prec,
                'recall': rec}

    def optimize(self, space, data_dict, max_evals=50, njobs=4, verbose=True):
        '''
        Wrapper method for using hyperopt (see utils.run_hyperopt
        for more details). After hyperparameter optimization, results
        are stored, the best model -overwrites- self.model, and the
        best params -overwrite- self.params.
        Inputs:
        space: a raytune compliant dictionary with defined optimization
            spaces. For example:
                space = {'max_iter'  : tune.qrandint(10, 10000, 10),
                        'tol'        : tune.loguniform(1e-6, 1e-4),
                        'gamma'      : tune.uniform(1, 50),
                        'n_neighbors': tune.qrandint(1, 200, 1)
                        }
            See hyperopt docs for more information.
        data_dict: compact data representation with the five requisite
            data structures used for training and testing an SSML model.
            keys trainx, trainy, testx, testy, and Ux required.
            NOTE: Uy is not needed since labels for unlabeled data
            instances is not used.
        max_evals: the number of epochs for hyperparameter optimization.
            Each iteration is one set of hyperparameters trained
            and tested on a fresh model. Convergence for simpler
            models like logistic regression typically happens well
            before 50 epochs, but can increase as more complex models,
            more hyperparameters, and a larger hyperparameter space is tested.
        njobs: (int) number of hyperparameter training iterations to complete
            in parallel. Default is 4, but personal computing resources may
            require less or allow more.
        verbose: boolean. If true, print results of hyperopt.
            If false, print only the progress bar for optimization.
        '''

        best, worst = run_hyperopt(space=space,
                                   model=self.fresh_start,
                                   data_dict=data_dict,
                                   max_evals=max_evals,
                                   njobs=njobs,
                                   verbose=verbose)

        # save the results of hyperparameter optimization
        self.best = best
        self.model = best['model']
        self.params = best['params']
        self.worst = worst

    def train(self, trainx, trainy, Ux):
        '''
        Wrapper method for sklearn's Label Propagation training method.
        Inputs:
        trainx: nxm feature vector/matrix for training model.
        trainy: nxk class label vector/matrix for training model.
        Ux: feature vector/matrix like labeled trainx but unlabeled data.
        '''

        # combine labeled and unlabeled instances for training
        lp_trainx = np.append(trainx, Ux, axis=0)
        lp_trainy = np.append(trainy,
                              np.full(shape=(Ux.shape[0],), fill_value=-1),
                              axis=0)

        # semi-supervised Label Propagation
        self.model.fit(lp_trainx, lp_trainy)

    def predict(self, testx, testy=None):
        '''
        Wrapper method for sklearn's Label Propagation predict method.
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
