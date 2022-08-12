import numpy as np
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from scripts.utils import STATUS_OK
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# shadow imports
import shadow.eaat
import shadow.losses
import shadow.utils
from shadow.utils import set_seed
# diagnostics
from scripts.utils import run_hyperopt
import joblib


class Net(nn.Module):
    '''
    Neural Network constructor .
    Also includes method for forward pass.
    nn.Module: PyTorch object for neural networks.
    Inputs:
    layer1: int length for first layer.
    layer2: int length for second layer.
        Ideally a multiple of layer1.
    layer3: int length for third layer.
        Ideally a multiple of layer2.
    kernel: convolutional kernel size.
        NOTE: An optimal value is unclear for spectral data.
    drop_rate: float (<1.) probability for reset/dropout layer.
    length: single instance data length.
        NOTE: Assumed to be 1000 for spectral data.
    TODO: Allow hyperopt to optimize on arbitrary sized networks.
    '''

    def __init__(self, layer1=32, layer2=64, layer3=128,
                 kernel=3, drop_rate=0.1, length=1000):
        '''
        Defines the structure for each type of layer.
        The resulting network has fixed length but the
        user can input arbitrary widths.
        '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, layer1, kernel, 1)
        self.conv2 = nn.Conv1d(layer1, layer2, kernel, 1)
        self.dropout = nn.Dropout2d(drop_rate)
        self.fc1 = nn.Linear(int(layer2*(length-2*(kernel-1))/2), layer3)
        # self.fc1 = nn.Linear(31744, 128)
        self.fc2 = nn.Linear(layer3, 2)

    def forward(self, x):
        '''
        The resulting network has a fixed length with
        two convolutional layers divided by relu activation,
        a max pooling layer, a dropout layer, and two
        fully-connected layers separated by a relu and
        dropout layers.
        '''

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SpectralDataset(torch.utils.data.Dataset):
    '''
    Dataset loader for use with PyTorch NN training.
    torch.utils.data.Dataset: managing user input data for random sampling.
    Inputs:
    trainD: the nxm input vector/matrix of data.
    labels: associated label vector for data.
    '''

    def __init__(self, trainD, labels):
        self.labels = labels
        self.trainD = trainD

    def __len__(self):
        '''
        Define what length is for the Dataset
        '''

        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Define how to retrieve an instance from a dataset.
        Inputs:
        idx: the index to sample from.
        '''

        label = self.labels[idx]
        data = self.trainD[idx]
        # no need to bother with labels, unpacking both anyways
        # sample = {"Spectrum": data, "Class": label}
        # return sample
        return data, label


class ShadowCNN:
    '''
    Methods for deploying a Shadow CNN
    implementation with hyperparameter optimization.
    Data agnostic (i.e. user supplied data inputs).
    TODO: Currently only supports binary classification.
        Add multinomial functions and unit tests.
        Add functionality for regression(?)
    Inputs:
    params: dictionary of logistic regression input functions.
        keys binning, hidden_layer, alpha, xi, eps, lr, and momentum
        are supported.
    TODO: Include functionality for manipulating other
        CNN architecture parameters in hyperparameter optimization
    random_state: int/float for reproducible intiailization.
    TODO: Add input parameter, loss_function, for the other
        loss function options available in Shadow (besides EAAT).
    '''

    # only binary so far
    def __init__(self, params=None, random_state=0):
        # defaults to a fixed value for reproducibility
        self.random_state = random_state
        # set seeds for reproducibility
        set_seed(0)
        # device used for computation
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        # dictionary of parameters for logistic regression model
        self.params = params
        if self.params is not None:
            # assumes the input dimensions are measurements of 1000 bins
            # TODO: Abstract this for arbitrary input size
            self.model = Net(layer1=params['layer1'],
                             layer2=2*params['layer1'],
                             layer3=3*params['layer1'],
                             kernel=params['kernel'],
                             drop_rate=params['drop_rate'],
                             length=1000)
            self.eaat = shadow.eaat.EAAT(model=self.model,
                                         alpha=params['alpha'],
                                         xi=params['xi'],
                                         eps=params['eps'])
            self.optimizer = optim.SGD(self.eaat.parameters(),
                                       lr=params['lr'],
                                       momentum=params['momentum'])
        else:
            # assumes the input dimensions are measurements of 1000 bins
            # TODO: Abstract this for arbitrary input size
            self.model = Net()
            self.eaat = shadow.eaat.EAAT(model=self.model)
            self.optimizer = optim.SGD(self.eaat.parameters())

    def fresh_start(self, params, data_dict):
        '''
        Required method for hyperopt optimization.
        Trains and tests a fresh Shadow NN model
        with given input parameters.
        This method does not overwrite self.model (self.optimize() does).
        Inputs:
        params: dictionary of logistic regression input functions.
            keys binning, layer1, alpha, xi, eps, lr, momentum,
            kernel, drop_rate, and batch_size are supported.
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

        # avoid float round-off by using DoubleTensor
        xtens = torch.FloatTensor(np.append(trainx,
                                            Ux,
                                            axis=0))[:, ::params['binning']]
        # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
        ytens = torch.LongTensor(np.append(trainy,
                                           np.full(shape=(Ux.shape[0],),
                                                   axis=0)))

        model = Net(layer1=params['layer1'],
                    layer2=2*params['layer1'],
                    layer3=3*params['layer1'],
                    kernel=params['kernel'],
                    drop_rate=params['drop_rate'],
                    length=xtens.shape[1])
        eaat = shadow.eaat.EAAT(model=model,
                                alpha=params['alpha'],
                                xi=params['xi'],
                                eps=params['eps'])
        optimizer = optim.SGD(eaat.parameters(),
                              lr=params['lr'],
                              momentum=params['momentum'])

        # define data set object
        dataset = SpectralDataset(xtens, ytens)

        # create DataLoader object of DataSet object
        DL_DS = torch.utils.data.DataLoader(dataset,
                                            batch_size=params['batch_size'],
                                            shuffle=True)

        # labels for unlabeled data are always "-1"
        xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1)

        n_epochs = 100
        eaat.to(self.device)
        losscurve = []
        evalcurve = []
        for epoch in range(n_epochs):
            eaat.train()
            lossavg = []
            for i, (data, targets) in enumerate(DL_DS):
                x = data.reshape((data.shape[0],
                                  1,
                                  data.shape[1])).to(self.device)
                y = targets.to(self.device)
                optimizer.zero_grad()
                out = eaat(x)
                loss = xEnt(out, y) + eaat.get_technique_cost(x)
                loss.backward()
                optimizer.step()
                lossavg.append(loss.item())
            losscurve.append(np.nanmedian(lossavg))
            evalcurve.append(self.predict(eaat,
                                          testx,
                                          testy,
                                          params['binning']))

        max_acc = np.max(evalcurve[-25:])

        return {'loss': 1-(max_acc/100.0),
                'status': STATUS_OK,
                'model': eaat,
                'params': params,
                'losscurve': losscurve,
                'evalcurve': evalcurve,
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
                space = {'layer1'        : scope.int(hp.quniform('layer1',
                                                                 1000,
                                                                 10000,
                                                                 10)),
                         'kernel'        : scope.int(hp.quniform('kernel',
                                                                 1,
                                                                 9,
                                                                 1)),
                         'alpha'        : hp.uniform('alpha', 0.0001, 0.999),
                         'xi'           : hp.uniform('xi', 1e-2, 1e0),
                         'eps'          : hp.uniform('eps', 0.5, 1.5),
                         'lr'           : hp.uniform('lr', 1e-3, 1e-1),
                         'momentum'     : hp.uniform('momentum', 0.5, 0.99),
                         'binning'      : scope.int(hp.quniform('binning',
                                                                1,
                                                                10,
                                                                1)),
                         'batch_szie'   : scope.int(hp.quniform('batch_size',
                                                                1,
                                                                100,
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
                                            axis=0))[:,
                                                     ::self.params['binning']]
        # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
        ytens = torch.LongTensor(np.append(trainy,
                                           np.full(shape=(Ux.shape[0],),
                                                   axis=0)))

        # define data set object
        dataset = SpectralDataset(xtens, ytens)

        # create DataLoader object of DataSet object
        DL_DS = torch.utils.data.DataLoader(dataset,
                                            batch_size=self.params[
                                                        'batch_size'
                                                        ],
                                            shuffle=True)

        # labels for unlabeled data are always "-1"
        xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1)

        n_epochs = 100
        self.eaat.to(self.device)
        losscurve = []
        evalcurve = []
        for epoch in range(n_epochs):
            self.eaat.train()
            lossavg = []
            for i, (data, targets) in enumerate(DL_DS):
                x = data.reshape((data.shape[0],
                                  1,
                                  data.shape[1])).to(self.device)
                y = targets.to(self.device)
                self.optimizer.zero_grad()
                out = self.eaat(x)
                loss = xEnt(out, y) + self.eaat.get_technique_cost(x)
                loss.backward()
                self.optimizer.step()
                lossavg.append(loss.item())
            losscurve.append(np.nanmedian(lossavg))
            evalcurve.append(self.predict(self.eaat,
                                          testx,
                                          testy,
                                          self.params['binning']))

        # optionally return the training accuracy if test data was provided
        return losscurve, evalcurve

    def predict(self, testx, testy=None, binning=1000):
        '''
        Wrapper method for Shadow NN predict method.
        Inputs:
        testx: nxm feature vector/matrix for testing model.
        testy: nxk class label vector/matrix for training model.
            optional: if included, the predicted classes -and-
            the resulting classification accuracy will be returned.
        '''

        self.eaat.eval()
        y_pred, y_true = [], []
        for i, data in enumerate(torch.FloatTensor(testx.copy()[:,
                                                                ::binning])):
            x = data.reshape((1, 1, data.shape[0])).to(self.device)
            out = self.eaat(x)
            y_pred.extend(torch.argmax(out, 1).detach().cpu().tolist())
        acc = None
        if testy is not None:
            y_true = torch.LongTensor(testy.copy())
            acc = (np.array(y_true) == np.array(y_pred)).mean() * 100

        return y_pred, acc

    def plot_cotraining(self, filename='lr-cotraining-learningcurves.png',
                        losscurve=None, evalcurve=None):
        '''
        Plots the training error curves for two co-training models.
        NOTE: The user can either choose to plot what is stored in
            the class instance by setting curves=None or
            the curves can be inputted.
        Inputs:
        filename: name to store picture under.
            Must end in .png (or will be added if missing).
        losscurve: the loss value over training epochs
        evalcurve: the accuracy scores over training epochs
        '''

        fig, (ax1, ax2) = plt.subplots(2,
                                       1,
                                       sharex=True,
                                       figsize=(10, 8),
                                       dpi=300)
        if losscurve is not None and evalcurve is not None:
            ax1.plot(losscurve)
            ax2.plot(evalcurve)
        else:
            ax1.plot(self.best['losscurve'])
            ax2.plot(self.best['evalcurve'])
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Curve')
        ax2.set_ylabel('Accuracy')
        ax1.grid()
        ax2.grid()

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