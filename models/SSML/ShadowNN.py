import numpy as np
# For hyperopt (parameter optimization)
from scripts.utils import STATUS_OK
# torch imports
import torch
# shadow imports
import shadow
# diagnostics
from scripts.utils import run_hyperopt
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
    random_state: int/float for reproducible intiailization.
    TODO: Add input parameter, loss_function, for the other
        loss function options available in Shadow (besides EAAT).
    '''

    # only binary so far
    def __init__(self, params=None, random_state=0):
        # defaults to a fixed value for reproducibility
        self.random_state = random_state
        # set seeds for reproducibility
        shadow.utils.set_seed(0)
        # device used for computation
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        # dictionary of parameters for logistic regression model
        self.params = params
        if self.params is not None:
            # assumes the input dimensions are measurements of 1000 bins
            # TODO: Abstract this for arbitrary input size
            self.eaat = shadow.eaat.EAAT(model=self.model_factory(
                                            1000//params['binning'],
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
            self.eaat_opt = torch.optim.SGD(self.eaat.parameters())
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

        eaat = shadow.eaat.EAAT(model=self.model_factory(
                                    testx[:, ::params['binning']].shape[1],
                                    params['hidden_layer']),
                                alpha=params['alpha'],
                                xi=params['xi'],
                                eps=params['eps']).to(self.device)
        eaat_opt = torch.optim.SGD(eaat.parameters(),
                                   lr=params['lr'],
                                   momentum=params['momentum'])
        xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

        # avoid float round-off by using DoubleTensor
        xtens = torch.FloatTensor(np.append(trainx,
                                            Ux,
                                            axis=0)[:, ::params['binning']])
        # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
        ytens = torch.LongTensor(np.append(trainy,
                                           np.full(shape=(Ux.shape[0],),
                                                   axis=0)))

        n_epochs = 100
        xt = torch.Tensor(xtens).to(self.device)
        yt = torch.LongTensor(ytens).to(self.device)
        # saves history for max accuracy
        acc_history = []
        # set the model into training mode
        # NOTE: change this to .eval() mode for testing and back again
        eaat.train()
        for epoch in range(n_epochs):
            # Forward/backward pass for training semi-supervised model
            out = eaat(xt)
            # supervised + unsupervised loss
            loss = xEnt(out, yt) + eaat.get_technique_cost(xt)
            eaat_opt.zero_grad()
            loss.backward()
            eaat_opt.step()

            eaat.eval()
            eaat_pred = torch.max(eaat(
                                    torch.FloatTensor(
                                        testx.copy()[:, ::params['binning']]
                                        )
                                    ), 1)[-1]
            acc = shadow.losses.accuracy(eaat_pred,
                                         torch.LongTensor(testy.copy())
                                         ).data.item()
            acc_history.append(acc)
        max_acc = np.max(acc_history[-20:])

        return {'loss': 1-(max_acc/100.0),
                'status': STATUS_OK,
                'model': eaat,
                'params': params,
                'accuracy': (max_acc/100.0)}

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
                space = {'hidden_layer' : scope.int(hp.quniform('hidden_layer',
                                                        1000,
                                                        10000,
                                                        10)),
                         'alpha'        : hp.uniform('alpha', 0.0001, 0.999),
                         'xi'           : hp.uniform('xi', 1e-2, 1e0),
                         'eps'          : hp.uniform('eps', 0.5, 1.5),
                         'lr'           : hp.uniform('lr', 1e-3, 1e-1),
                         'momentum'     : hp.uniform('momentum', 0.5, 0.99),
                         'binning'      : scope.int(hp.quniform('binning',
                                                                1,
                                                                10,
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
                                                   axis=0)))

        n_epochs = 100
        xt = torch.Tensor(xtens).to(self.device)
        yt = torch.LongTensor(ytens).to(self.device)
        # saves history for max accuracy
        acc_history = []
        # set the model into training mode
        # NOTE: change this to .eval() mode for testing and back again
        self.eaat.train()
        for epoch in range(n_epochs):
            # Forward/backward pass for training semi-supervised model
            out = self.eaat(xt)
            # supervised + unsupervised loss
            loss = self.xEnt(out, yt) + self.eaat.get_technique_cost(xt)
            self.eaat_opt.zero_grad()
            loss.backward()
            self.eaat_opt.step()

            if testx is not None and testy is not None:
                self.eaat.eval()
                eaat_pred = torch.max(self.eaat(
                                        torch.FloatTensor(
                                            testx.copy()[:,
                                                         ::self.params[
                                                            'binning']
                                                         ]
                                            )
                                        ), 1)[-1]
                acc = shadow.losses.accuracy(eaat_pred,
                                             torch.LongTensor(testy.copy())
                                             ).data.item()
                acc_history.append(acc)

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
                                    )
                                ), 1)[-1]

        acc = None
        if testy is not None:
            acc = shadow.losses.accuracy(eaat_pred,
                                         torch.LongTensor(testy.copy())
                                         ).data.item()

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
