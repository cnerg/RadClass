import numpy as np
# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# torch imports
import torch
# shadow imports
import shadow.eaat
import shadow.losses
import shadow.utils
from shadow.utils import set_seed
# diagnostics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from scripts.utils import EarlyStopper, run_hyperopt
import joblib


class ShadowNN:
    '''
    Methods for deploying a Shadow fully-connected NN
    implementation with hyperparameter optimization.
    Data agnostic (i.e. user supplied data inputs).
    TODO: Currently only supports binary classification.
        Add multinomial functions and unit tests.
        Add functionality for regression(?)
    Inputs:
    params: dictionary of logistic regression input functions.
        keys binning, hidden_layer, alpha, xi, eps, lr, and momentum
        are supported.
    alpha: float; weight for encouraging high recall
    beta: float; weight for encouraging high precision
    NOTE: if alpha=beta=0, default to favoring balanced accuracy.
    random_state: int/float for reproducible intiailization.
    TODO: Add input parameter, loss_function, for the other
        loss function options available in Shadow (besides EAAT).
    '''

    # only binary so far
    def __init__(self, params=None, alpha=0, beta=0, random_state=0, input_length=1000):
        # defaults to a fixed value for reproducibility
        self.random_state = random_state
        self.input_length = input_length
        # set seeds for reproducibility
        set_seed(0)
        # device used for computation
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        # dictionary of parameters for neural network model
        self.alpha, self.beta = alpha, beta
        self.params = params
        if self.params is not None:
            # assumes the input dimensions are measurements of 1000 bins
            # TODO: Abstract this for arbitrary input size
            self.eaat = shadow.eaat.EAAT(model=self.model_factory(
                                            int(np.ceil(
                                                self.input_length /
                                                params['binning'])),
                                            params['hidden_layer']),
                                         alpha=params['alpha'],
                                         xi=params['xi'],
                                         eps=params['eps']).to(self.device)
            self.eaat_opt = torch.optim.SGD(self.eaat.parameters(),
                                            lr=params['lr'],
                                            momentum=params['momentum'])
            # unlabeled instances always have a label of "-1"
            self.xEnt = torch.nn.CrossEntropyLoss(
                            ignore_index=-1).to(self.device)
        else:
            self.params = {'binning': 1}
            # assumes the input dimensions are measurements of 1000 bins
            self.eaat = shadow.eaat.EAAT(
                            model=self.model_factory()).to(self.device)
            self.eaat_opt = torch.optim.SGD(self.eaat.parameters(),
                                            lr=0.1, momentum=0.9)
            # unlabeled instances always have a label of "-1"
            self.xEnt = torch.nn.CrossEntropyLoss(
                            ignore_index=-1).to(self.device)

    def model_factory(self, length=1000, hidden_layer=10000):
        return torch.nn.Sequential(
            torch.nn.Linear(length, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, length),
            torch.nn.ReLU(),
            torch.nn.Linear(length, 2)
        )

    def fresh_start(self, params, data_dict):
        '''
        Required method for hyperopt optimization.
        Trains and tests a fresh Shadow NN model
        with given input parameters.
        This method does not overwrite self.model (self.optimize() does).
        Inputs:
        params: dictionary of logistic regression input functions.
            keys binning, hidden_layer, alpha, xi, eps, lr, and momentum
            are supported.
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

        clf = ShadowNN(params=params,
                       random_state=self.random_state,
                       input_length=testx.shape[1])
        # training and testing
        acc_history = clf.train(trainx, trainy, Ux, testx, testy)
        # not used; max acc in past few epochs used instead
        eaat_pred, acc = clf.predict(testx, testy)
        max_acc = np.max(acc_history[-10:])
        rec = recall_score(testy, eaat_pred)
        prec = precision_score(testy, eaat_pred)
        score = max_acc+(self.alpha*rec)+(self.beta*prec)
        loss = (1-max_acc) + self.alpha*(1-rec)+self.beta*(1-prec)

        # loss function minimizes misclassification
        # by maximizing metrics
        return {'score': score,
                'loss': loss,
                'model': clf.eaat,
                'params': params,
                'accuracy': max_acc,
                'precision': prec,
                'recall': rec,
                'evalcurve': acc_history}

    def optimize(self, space, data_dict, max_evals=50, njobs=4, verbose=True):
        '''
        Wrapper method for using hyperopt (see utils.run_hyperopt
        for more details). After hyperparameter optimization, results
        are stored, the best model -overwrites- self.model, and the
        best params -overwrite- self.params.
        Inputs:
        space: a raytune compliant dictionary with defined optimization
            spaces. For example:
                space = {'hidden_layer' : tune.qrandint(1000, 10000, 10),
                         'alpha'        : tune.uniform(0.0001, 0.999),
                         'xi'           : tune.uniform(1e-2, 1e0),
                         'eps'          : tune.uniform(0.5, 1.5),
                         'lr'           : tune.uniform(1e-3, 1e-1),
                         'momentum'     : tune.uniform(0.5, 0.99),
                         'binning'      : tune.qrandint(1, 10, 1)
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

    def train(self, trainx, trainy, Ux, testx=None, testy=None):
        '''
        Wrapper method for Shadow NN training method.
        Inputs:
        trainx: nxm feature vector/matrix for training model.
        trainy: nxk class label vector/matrix for training model.
        Ux: feature vector/matrix like labeled trainx but unlabeled data.
        testx: feature vector/matrix used for testing the performance
            of each model at every iteration.
        testy: label vector used for testing the performance
            of each model at every iteration.
        '''

        # avoid float round-off by using DoubleTensor
        xtens = torch.FloatTensor(np.append(trainx,
                                            Ux,
                                            axis=0)[:,
                                                    ::self.params['binning']])
        # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
        ytens = torch.LongTensor(np.append(trainy,
                                           np.full(shape=(Ux.shape[0],),
                                                   fill_value=-1),
                                           axis=0))

        n_epochs = 100
        xt = torch.Tensor(xtens).to(self.device)
        yt = torch.LongTensor(ytens).to(self.device)
        # generate early-stopping watchdog
        # TODO: allow a user of ShadowCNN to specify EarlyStopper's params
        stopper = EarlyStopper(patience=3, min_delta=0)
        # saves history for max accuracy
        acc_history = []
        for epoch in range(n_epochs):
            # set the model into training mode
            # NOTE: change this to .eval() mode for testing and back again
            self.eaat.train()
            # Forward/backward pass for training semi-supervised model
            out = self.eaat(xt)
            # supervised + unsupervised loss
            loss = self.xEnt(out, yt) + self.eaat.get_technique_cost(xt)
            self.eaat_opt.zero_grad()
            loss.backward()
            self.eaat_opt.step()

            if testx is not None and testy is not None:
                x_val = torch.FloatTensor(
                            testx.copy()
                        )[:, ::self.params['binning']].to(self.device)
                y_val = torch.LongTensor(testy.copy()).to(self.device)

                self.eaat.eval()
                eaat_pred = torch.max(self.eaat(x_val), 1)[-1]
                acc = balanced_accuracy_score(testy, eaat_pred.cpu().numpy())
                acc_history.append(acc)

                self.eaat.train()
                # test for early stopping
                out = self.eaat(x_val)
                val_loss = self.xEnt(out, y_val) + \
                    self.eaat.get_technique_cost(x_val)
                if stopper.early_stop(val_loss):
                    break

        # optionally return the training accuracy if test data was provided
        return acc_history

    def predict(self, testx, testy=None):
        '''
        Wrapper method for Shadow NN predict method.
        Inputs:
        testx: nxm feature vector/matrix for testing model.
        testy: nxk class label vector/matrix for training model.
            optional: if included, the predicted classes -and-
            the resulting classification accuracy will be returned.
        '''

        self.eaat.eval()
        eaat_pred = torch.max(self.eaat(
                                torch.FloatTensor(
                                    testx.copy()[:, ::self.params['binning']]
                                    ).to(self.device)
                                ), 1)[-1]
        # return tensor to cpu if on gpu and convert to numpy for return
        eaat_pred = eaat_pred.cpu().numpy()

        acc = None
        if testy is not None:
            acc = balanced_accuracy_score(testy, eaat_pred)

        return eaat_pred, acc

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
