import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
# For hyperparameter optimization
from hyperopt import Trials, tpe, fmin
from functools import partial
# diagnostics
from sklearn.metrics import confusion_matrix
# pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Cross Validation
from sklearn.model_selection import KFold, StratifiedKFold


class EarlyStopper:
    '''
    Early stopping mechanism for neural networks.
    Code adapted from user "isle_of_gods" from StackOverflow:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    Use this class to break a training loop if the validation loss is low.
    Inputs:
    patience: integer; forces stop if validation loss has not improved
        for some time
    min_delta: "fudge value" for how much loss to tolerate before stopping
    '''

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        '''
        Tests for the early stopping condition if the validation loss
        has not improved for a certain period of time (patience).
        Inputs:
        validation_loss: typically a float value for the loss function of
            a neural network training loop
        '''

        if validation_loss < self.min_validation_loss:
            # keep track of the smallest validation loss
            # if it has been beaten, restart patience
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # keep track of whether validation loss has been decreasing
            # by a tolerable amount
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def run_hyperopt(space, model, data, testset, metric='loss', mode='min',
                 max_evals=50, verbose=True):
    '''
    Runs hyperparameter optimization on a model given a parameter space.
    Inputs:
    space: dictionary with each hyperparameter as keys and values being the
        range of parameter space (see hyperopt docs for defining a space)
    mode: function that takes params dictionary, trains a specified ML model
        and returns the optimization loss function, model, and other
        attributes (e.g. accuracy on evaluation set)
    max_eval: (int) run hyperparameter optimization for max_val iterations
    njobs: (int) number of hyperparameter training iterations to complete
        in parallel. Default is 4, but personal computing resources may
        require less or allow more.
    verbose: report best and worse loss/accuracy

    Returns:
    best: dictionary with returns from model function, including best loss,
        best trained model, best parameters, etc.
    '''

    # algo = HyperOptSearch(metric=metric, mode=mode)
    # algo = ConcurrencyLimiter(algo, max_concurrent=njobs)

    trials = Trials()

    # wrap data into objective function
    fmin_objective = partial(model, data=data, testset=testset)

    # trainable = tune.with_resources(
    #     tune.with_parameters(model, data=data, testset=testset),
    #     {"cpu": num_workers+1})
    # run hyperopt
    # tuner = tune.Tuner(
    #             trainable,
    #             param_space=space,
    #             # run_config=air.RunConfig(stop=TimeStopper()),
    #             tune_config=tune.TuneConfig(num_samples=max_evals,
    #                                         metric=metric,
    #                                         mode=mode,
    #                                         search_alg=algo),
    #                                         # time_budget_s=3600),
    #         )

    # results = tuner.fit()

    fmin(fmin_objective,
         space,
         algo=tpe.suggest,
         max_evals=max_evals,
         trials=trials)

    # of all trials, find best and worst loss/accuracy from optimization
    # if mode == 'min':
    #     worst_mode = 'max'
    # else:
    #     worst_mode = 'min'
    # best = results.get_best_result(metric=metric, mode=mode)
    # worst = results.get_best_result(metric=metric, mode=worst_mode)
    best = trials.results[np.argmin([r['loss'] for r in trials.results])]
    worst = trials.results[np.argmax([r['loss'] for r in trials.results])]
    # best_checkpoint = best.checkpoint
    # best = best.metrics
    # worst_checkpoint = worst.checkpoint
    # worst = worst.metrics

    if verbose:
        print('best metrics:')
        print('\taccuracy:', best['accuracy'])
        # print('\tprecision:', best['precision'])
        # print('\trecall:', best['recall'])
        # print('\tscore:', best['score'])
        print('\tloss:', best['loss'])
        print('\tparams:', best['params'])
        # print('\tmodel:', best['model'])

        print('worst metrics:')
        print('\taccuracy:', worst['accuracy'])
        # print('\tprecision:', worst['precision'])
        # print('\trecall:', worst['recall'])
        # print('\tscore:', worst['score'])
        print('\tloss:', worst['loss'])
        print('\tparams:', worst['params'])
        # print('\tmodel:', worst['model'])

    return best, worst, trials  # , best_checkpoint, worst_checkpoint


def cross_validation(model, X, y, params, n_splits=3,
                     stratified=False, random_state=None):
    '''
    Perform K-Fold cross validation using sklearn and a given model.
    The model *must* have a fresh_start method (see models in RadClass/models).
    fresh_start() is used instead of train() to be agnostic to the data needed
        for training (fresh_start requires a data_dict whereas each model's
        train could take different combinations of labeled & unlabeled data).
        This also avoids the need to do hyperparameter optimization (and
        therefore many training epochs) for every K-Fold.
    NOTE: fresh_start returns the model and results in a dictionary but
        does not overwrite/save the model to the respective class.
        You can manually overwrite using model.model = return.model
    Hyperparameter optimization (model.optimize) can be done before or after
        cross validation to specify the (optimal) parameters used by the model
        since they are required here.
    NOTE: Fixed default to shuffle data during cross validation splits.
        (See sklearn cross validation docs for more info.)
    NOTE: Unlabeled data, if provided, will always be included in the training
        dataset. This means that this cross validation implementation is
        susceptible to bias in the unlabeled data distribution. To test for
        this bias, a user can manually run cross validation as a parent to
        calling this function, splitting the unlabeled data and adding
        different folds into X.
    Inputs:
    model: ML model class object (e.g. RadClass/models).
        Must have a fresh_start() method.
        NOTE: If the model expects unlabeled data but unlabed data is not
        provided in X/y, an error will likely be thrown when training the model
        through fresh_start.
    X: array of feature vectors (rows of individual instances, cols of vectors)
        This should include all data for training and testing (since the
        testing subset will be split by cross validation), including unlabeled
        data if needed/used.
    y: array/vector of labels for X. If including unlabeled data, use -1.
        This should have the same order as X. That is, each row index in X
        has an associated label with the same index in y.
    params: dictionary of hyperparameters. Will depend on model used.
        Alternatively, use model.params for models in RadClass/models
    n_splits: int number of splits for K-Fold cross validation
    stratified: bool; if True, balance the K-Folds to have roughly the same
        proportion of samples from each class.
    random_state: seed for reproducility.
    '''

    # return lists
    accs = []
    reports = []

    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                             shuffle=True)
    else:
        cv = KFold(n_splits=n_splits, random_state=random_state,
                   shuffle=True)

    # separate unlabeled data if included
    Ux = None
    Uy = None
    if -1 in y:
        U_idx = np.where(y == -1)[0]
        L_idx = np.where(y != -1)[0]
        Ux = X[U_idx]
        Uy = y[U_idx]
        Lx = X[L_idx]
        Ly = y[L_idx]
    else:
        Lx = X
        Ly = y
    # conduct K-Fold cross validation
    cv.get_n_splits(Lx, Ly)
    for train_idx, test_idx in cv.split(Lx, Ly):
        trainx, testx = Lx[train_idx], Lx[test_idx]
        trainy, testy = Ly[train_idx], Ly[test_idx]

        # construct data dictionary for training in fresh_start
        data_dict = {'trainx': trainx, 'trainy': trainy,
                     'testx': testx, 'testy': testy}
        if Ux is not None:
            data_dict['Ux'] = Ux
            data_dict['Uy'] = Uy
        results = model.fresh_start(params, data_dict)
        accs = np.append(accs, results['accuracy'])
        reports = np.append(reports, results)

    # report cross validation results
    print('Average accuracy:', np.mean(accs))
    print('Max accuracy:', np.max(accs))
    print('All accuracy:', accs)
    # return the results of fresh_start for the max accuracy model
    return reports[np.argmax(accs)]


def pca(Lx, Ly, Ux, Uy, filename):
    '''
    A function for computing and plotting 2D PCA.
    Inputs:
    Lx: labeled feature data.
    Ly: class labels for labeled data.
    Ux: unlabeled feature data.
    Uy: labels for unlabeled data (all labels should be -1).
    filename: filename for saved plot.
        The file must be saved with extension .joblib.
        Added to filename if not included as input.
    '''

    plt.rcParams.update({'font.size': 20})
    # only saving colors for binary classification with unlabeled instances
    col_dict = {-1: 'tab:gray', 0: 'tab:orange', 1: 'tab:blue'}

    pcadata = np.append(Lx, Ux, axis=0)
    normalizer = StandardScaler()
    x = normalizer.fit_transform(pcadata)
    print(np.mean(pcadata), np.std(pcadata))
    print(np.mean(x), np.std(x))

    pca = PCA(n_components=2)
    pca.fit_transform(x)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.components_)

    principalComponents = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    for idx, color in col_dict.items():
        indices = np.where(np.append(Ly, Uy, axis=0) == idx)[0]
        ax.scatter(principalComponents[indices, 0],
                   principalComponents[indices, 1],
                   c=color,
                   label='class '+str(idx))
    ax.grid()
    ax.legend()

    if filename[-4:] != '.png':
        filename += '.png'
    fig.tight_layout()
    fig.savefig(filename)


def multiD_pca(Lx, Ly, Ux, Uy, filename, n=2):
    '''
    A function for computing and plotting n-dimensional PCA.
    Inputs:
    Lx: labeled feature data.
    Ly: class labels for labeled data.
    Ux: unlabeled feature data.
    Uy: labels for unlabeled data (all labels should be -1).
    filename: filename for saved plot.
        The file must be saved with extension .joblib.
        Added to filename if not included as input.
    n: number of singular values to include in PCA analysis.
    '''

    plt.rcParams.update({'font.size': 20})
    # only saving colors for binary classification with unlabeled instances
    col_dict = {-1: 'tab:gray', 0: 'tab:orange', 1: 'tab:blue'}

    pcadata = np.append(Lx, Ux, axis=0)
    normalizer = StandardScaler()
    x = normalizer.fit_transform(pcadata)
    print(np.mean(pcadata), np.std(pcadata))
    print(np.mean(x), np.std(x))

    n = 2
    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.components_)

    alph = ["A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J", "K", "L", "M", "N", "O", "P",
            "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z"]
    jobs = alph[:n]

    fig, axes = plt.subplots(n, n, figsize=(15, 15))

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if row == col:
                ax.tick_params(
                    axis='both', which='both',
                    bottom='off', top='off',
                    labelbottom='off',
                    left='off', right='off',
                    labelleft='off'
                )
                ax.text(0.5, 0.5, jobs[row], horizontalalignment='center')
            else:
                for idx, color in col_dict.items():
                    indices = np.where(np.append(Ly, Uy, axis=0) == idx)[0]
                    ax.scatter(principalComponents[indices, row],
                               principalComponents[indices, col],
                               c=color,
                               label='class '+str(idx))
    fig.tight_layout()
    if filename[-4:] != '.png':
        filename += '.png'
    fig.savefig(filename)


def plot_cf(testy, predy, title, filename):
    '''
    Uses sklearn metric to compute a confusion matrix for visualization
    Inputs:
    testy: array/vector with ground-truth labels for test/evaluation set
    predy: array/vector with predicted sample labels from trained model
    title: string title for plot
    filename: string with extension for confusion matrix file
    '''

    cf_matrix = confusion_matrix(testy, predy)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0(SNM)', '1(other)'])
    ax.yaxis.set_ticklabels(['0(SNM)', '1(other)'])
    # Save the visualization of the Confusion Matrix.
    plt.savefig(filename)
