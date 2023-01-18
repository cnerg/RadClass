import numpy as np
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# sklearn models
from sklearn import linear_model
# diagnostics
from sklearn.metrics import balanced_accuracy_score
from scripts.utils import run_hyperopt
import joblib


class CoTraining:
    '''
    Methods for deploying a basic co-training with logistic
    regression implementation with hyperparameter optimization.
    Data agnostic (i.e. user supplied data inputs).
    TODO: Currently only supports binary classification.
        - Add multinomial functions and unit tests.
        - Add functionality for regression(?)
    Inputs:
    kwargs: logistic regression input functions.
        keys seed, random_state, max_iter, tol, and C supported.
        seed/random_state: int/float for reproducible intiailization.
    '''

    # only binary so far
    def __init__(self, **kwargs):
        # supported keys = ['max_iter', 'tol', 'C', 'random_state', 'seed']
        # defaults to a fixed value for reproducibility
        self.random_state = kwargs.pop('random_state', 0)
        # set the random seed of training splits for reproducibility
        self.seed = kwargs.pop('seed', 0)
        np.random.seed(self.seed)
        # parameters for cotraining logistic regression models:
        # defaults to sklearn.linear_model.LogisticRegression default vals
        self.max_iter = kwargs.pop('max_iter', 100)
        self.tol = kwargs.pop('tol', 0.0001)
        self.C = kwargs.pop('C', 1.0)
        self.n_samples = kwargs.pop('n_samples', 1)
        self.model1 = linear_model.LogisticRegression(
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        C=self.C
                    )
        self.model2 = linear_model.LogisticRegression(
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        C=self.C
                    )

    def training_loop(self, slr1, slr2, L_lr1, L_lr2,
                      Ly_lr1, Ly_lr2, U_lr, n_samples,
                      testx=None, testy=None):
        '''
        Main training iteration for co-training.
        Given two models, labeled training data, and unlabeled training data:
        - Train both models using their respective labeled datasets
        - Randomly sample n_samples number of unlabeled
            instances for model 1 and 2 each.
        - Label the sampled unlabeled instances using
            model 1 (u1) and model 2 (u2).
        - Remove u1 and u2 from the unlabeled dataset and
            include in each model's respective labeled dataset
            with their associated labels for future training.
        Inputs:
        slr1: logistic regression co-training model #1
        slr2: logistic regression co-training model #2
        L_lr1: feature training data for co-training model #1
        L_lr2: feature training data for co-training model #2
        Ly_lr1: labels for input data for co-training model #1
        Ly_lr2: labels for input data for co-training model #2
        U_lr: unlabeled feature training data used by both models
        n_samples: the number of instances to sample and
            predict from Ux at one time
        testx: feature vector/matrix used for testing the performance
            of each model at every iteration.
        testy: label vector used for testing the performance
            of each model at every iteration.
        '''

        model1_accs, model2_accs = np.array([]), np.array([])
        # should stay false but if true,
        # the same unalbeled instance could be sampled multiple times
        rep = False
        while U_lr.shape[0] > 1:
            slr1.fit(L_lr1, Ly_lr1)
            slr2.fit(L_lr2, Ly_lr2)

            # pull u1
            # ensuring there is enough instances to sample for each model
            if U_lr.shape[0] < n_samples*2:
                n_samples = int(U_lr.shape[0]/2)
            uidx1 = np.random.choice(range(U_lr.shape[0]),
                                     n_samples,
                                     replace=rep)
            u1 = U_lr[uidx1].copy()
            # remove instances that will be labeled
            U_lr = np.delete(U_lr, uidx1, axis=0)

            # pull u2
            uidx2 = np.random.choice(range(U_lr.shape[0]),
                                     n_samples,
                                     replace=rep)
            u2 = U_lr[uidx2].copy()
            # remove instances that will be labeled
            U_lr = np.delete(U_lr, uidx2, axis=0)

            # predict unlabeled samples
            u1y = slr1.predict(u1)
            u2y = slr2.predict(u2)

            if testx is not None and testy is not None:
                # test and save model(s) accuracy over all training iterations
                model1_accs = np.append(model1_accs,
                                        balanced_accuracy_score(testy,
                                                                slr1.predict(
                                                                    testx)))
                model2_accs = np.append(model2_accs,
                                        balanced_accuracy_score(testy,
                                                                slr2.predict(
                                                                    testx)))

            # add predictions to cotrained model(s) labeled samples
            L_lr1 = np.append(L_lr1, u2, axis=0)
            L_lr2 = np.append(L_lr2, u1, axis=0)
            Ly_lr1 = np.append(Ly_lr1, u2y, axis=0)
            Ly_lr2 = np.append(Ly_lr2, u1y, axis=0)

        return slr1, slr2, model1_accs, model2_accs

    def fresh_start(self, params, data_dict):
        '''
        Required method for hyperopt optimization.
        Trains and tests a fresh co-training model
        with given input parameters.
        This method does not overwrite self.model (self.optimize() does).
        Inputs:
        params: dictionary of logistic regression input functions.
            keys n_samples, max_iter, tol, and C supported.
        data_dict: compact data representation with the four requisite
            data structures used for training and testing a model.
            keys trainx, trainy, testx, testy, and Ux required.
            NOTE: Uy is not needed since labels for unlabeled data
            instances is not used.
        '''

        # unpack data
        trainx = data_dict['trainx']
        trainy = data_dict['trainy']
        testx = data_dict['testx']
        testy = data_dict['testy']
        # unlabeled co-training data
        Ux = data_dict['Ux']

        clf = CoTraining(**params, random_state=self.random_state)
        # training and testing
        model1_accs, model2_accs = clf.train(trainx, trainy, Ux, testx, testy)
        # uses balanced_accuracy accounts for class imbalanced data
        pred1, acc, pred2, model1_acc, model2_acc = clf.predict(testx, testy)

        return {'loss': 1-acc,
                'status': STATUS_OK,
                'model': clf.model1,
                'model2': clf.model2,
                'model1_acc_history': model1_accs,
                'model2_acc_history': model2_accs,
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
                space = {'max_iter' : scope.int(hp.quniform('max_iter',
                                                            10,
                                                            10000,
                                                            10)),
                        'tol'       : hp.loguniform('tol', 1e-5, 1e-3),
                        'C'         : hp.uniform('C', 1.0, 1000.0),
                        'n_samples' : scope.int(hp.quniform('n_samples',
                                                            1,
                                                            20,
                                                            1))
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

    def train(self, trainx, trainy, Ux,
              testx=None, testy=None):
        '''
        Wrapper method for a basic co-training with logistic regression
        implementation training method.
        Inputs:
        trainx: nxm feature vector/matrix for training model.
        trainy: nxk class label vector/matrix for training model.
        Ux: feature vector/matrix like labeled trainx but unlabeled data.
        testx: feature vector/matrix used for testing the performance
            of each model at every iteration.
        testy: label vector used for testing the performance
            of each model at every iteration.
        '''

        # avoid overwriting when deleting in co-training loop
        U_lr = Ux.copy()

        # TODO: allow a user to specify uneven splits between the two models
        split_frac = 0.5
        # labeled training data
        idx = np.random.choice(range(trainy.shape[0]),
                               size=int(split_frac * trainy.shape[0]),
                               replace=False)

        # avoid overwriting when deleting in co-training loop
        L_lr1 = trainx[idx].copy()
        L_lr2 = trainx[~idx].copy()
        Ly_lr1 = trainy[idx].copy()
        Ly_lr2 = trainy[~idx].copy()

        self.model1, self.model2, model1_accs, model2_accs = \
            self.training_loop(
                                self.model1, self.model2,
                                L_lr1, L_lr2,
                                Ly_lr1, Ly_lr2,
                                U_lr, self.n_samples,
                                testx, testy,
                                )

        # optional returns if a user is interested in training diagnostics
        return model1_accs, model2_accs

    def predict(self, testx, testy=None):
        '''
        Wrapper method for sklearn's Label Propagation predict method.
        Inputs:
        testx: nxm feature vector/matrix for testing model.
        testy: nxk class label vector/matrix for training model.
            optional: if included, the predicted classes -and-
            the resulting classification accuracy will be returned.
        '''

        pred1 = self.model1.predict(testx)
        pred2 = self.model2.predict(testx)

        acc = None
        if testy is not None:
            # balanced_accuracy accounts for class imbalanced data
            # could alternatively use pure accuracy
            # for a more traditional hyperopt
            model1_acc = balanced_accuracy_score(testy, pred1)
            model2_acc = balanced_accuracy_score(testy, pred2)
            # select best accuracy for hyperparameter optimization
            acc = max(model1_acc, model2_acc)

        return pred1, acc, pred2, model1_acc, model2_acc

    def plot_cotraining(self, model1_accs=None, model2_accs=None,
                        filename='lr-cotraining-learningcurves.png'):
        '''
        Plots the training error curves for two co-training models.
        NOTE: The user must provide the curves to plot, but each curve is
            saved by the class under self.best and self.worst models.
        Inputs:
        filename: name to store picture under.
            Must end in .png (or will be added if missing).
        model1_accs: the accuracy scores over training epochs for model 1
        model2_accs: the accuracy scores over training epochs for model 2
        '''

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.plot(np.arange(len(model1_accs)), model1_accs,
                color='tab:blue', label='Model 1')
        ax.plot(np.arange(len(model2_accs)), model2_accs,
                color='tab:orange', label='Model 2')
        ax.legend()
        ax.set_xlabel('Co-Training Iteration')
        ax.set_ylabel('Test Accuracy')
        ax.grid()

        if filename[-4:] != '.png':
            filename += '.png'
        fig.savefig(filename)

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
