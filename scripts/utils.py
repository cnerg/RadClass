import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from hyperopt import Trials, tpe, fmin
from functools import partial
# diagnostics
from sklearn.metrics import confusion_matrix
import logging
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
        return self.counter >= self.patience


def run_hyperopt(space, model, data_dict, max_evals=50, verbose=True):
    '''
    Runs hyperparameter optimization on a model given a parameter space.
    Inputs:
    space: dictionary with each hyperparameter as keys and values being the
        range of parameter space (see hyperopt docs for defining a space)
    mode: function that takes params dictionary, trains a specified ML model
        and returns the optimization loss function, model, and other
        attributes (e.g. accuracy on evaluation set)
    max_eval: (int) run hyperparameter optimization for max_val iterations
    verbose: report best and worse loss/accuracy

    Returns:
    best: dictionary with returns from model function, including best loss,
        best trained model, best parameters, etc.
    worst: dictionary with returns from model function, including worst loss,
        worst trained model, worst parameters, etc.
    '''

    trials = Trials()

    # wrap data into objective function
    fmin_objective = partial(model, data_dict=data_dict)

    # run hyperopt
    fmin(fmin_objective,
         space,
         algo=tpe.suggest,
         max_evals=max_evals,
         trials=trials)

    # of all trials, find best and worst loss/accuracy from optimization
    best = trials.results[np.argmin([r['loss'] for r in trials.results])]
    worst = trials.results[np.argmax([r['loss'] for r in trials.results])]

    if verbose:
        logging.info('best accuracy:', 1-best['loss'])
        logging.info('best params:', best['params'])
        logging.info('worst accuracy:', 1-worst['loss'])
        logging.info('worst params:', worst['params'])

    return best, worst


def cross_validation(model, X, y, params, n_splits=3,
                     stratified=False, random_state=None, shuffle=True):
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
    shuffle: bool; if True, shuffle the data before conducting K-folds.
    '''

    # return lists
    accs = []
    reports = []

    if stratified:  # pragma: no cover
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                             shuffle=shuffle)
    else:   # pragma: no cover
        cv = KFold(n_splits=n_splits, random_state=random_state,
                   shuffle=shuffle)

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
    logging.info('Average accuracy:', np.mean(accs))
    logging.info('Max accuracy:', np.max(accs))
    logging.info('All accuracy:', accs)
    # return the results of fresh_start for the max accuracy model
    return reports


def pca(Lx, Ux, n):  # pragma: no cover
    '''
    A function for computing n-component PCA.
    Inputs:
    Lx: labeled feature data.
    Ux: unlabeled feature data.
    n: number of singular values to include in PCA analysis.
    '''

    pcadata = np.append(Lx, Ux, axis=0)
    normalizer = StandardScaler()
    x = normalizer.fit_transform(pcadata)
    logging.info(np.mean(pcadata), np.std(pcadata))
    logging.info(np.mean(x), np.std(x))

    pca = PCA(n_components=n)
    pca.fit_transform(x)
    logging.info(pca.explained_variance_ratio_)
    logging.info(pca.singular_values_)
    logging.info(pca.components_)

    principalComponents = pca.fit_transform(x)

    return principalComponents


def _plot_pca_data(principalComponents, Ly, Uy, ax):    # pragma: no cover
    '''
    Helper function for plot_pca that plots data for a given axis.
    Inputs:
    principalComponents: ndarray of shape (n_samples, n_components).
    Ly: class labels for labeled data.
    Uy: labels for unlabeled data (all labels should be -1).
    ax: matplotlib-axis to plot on.
    '''

    # only saving colors for binary classification with unlabeled instances
    col_dict = {-1: 'tab:gray', 0: 'tab:orange', 1: 'tab:blue'}

    for idx, color in col_dict.items():
        indices = np.where(np.append(Ly, Uy, axis=0) == idx)[0]
        ax.scatter(principalComponents[indices, 0],
                   principalComponents[indices, 1],
                   c=color,
                   label='class '+str(idx))
    return ax


def plot_pca(principalComponents, Ly, Uy, filename, n=2):   # pragma: no cover
    '''
    A function for computing and plotting n-dimensional PCA.
    Inputs:
    principalComponents: ndarray of shape (n_samples, n_components).
    Ly: class labels for labeled data.
    Uy: labels for unlabeled data (all labels should be -1).
    filename: filename for saved plot.
        The file must be saved with extension .joblib.
        Added to filename if not included as input.
    n: number of singular values to include in PCA analysis.
    '''

    plt.rcParams.update({'font.size': 20})

    alph = ["A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J", "K", "L", "M", "N", "O", "P",
            "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z"]
    jobs = alph[:n]

    # only one plot is needed for n=2
    if n == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel('PC '+jobs[0], fontsize=15)
        ax.set_ylabel('PC '+jobs[1], fontsize=15)
        ax = _plot_pca_data(principalComponents, Ly, Uy, ax)
        ax.grid()
        ax.legend()
    else:
        fig, axes = plt.subplots(n, n, figsize=(15, 15))
        for row in range(axes.shape[0]):
            for col in range(axes.shape[1]):
                ax = axes[row, col]
                # blank label plot
                if row == col:
                    ax.tick_params(
                        axis='both', which='both',
                        bottom='off', top='off',
                        labelbottom='off',
                        left='off', right='off',
                        labelleft='off'
                    )
                    ax.text(0.5, 0.5, jobs[row], horizontalalignment='center')
                # PCA results
                else:
                    ax = _plot_pca_data(principalComponents, Ly, Uy, ax)

    if filename[-4:] != '.png':
        filename += '.png'
    fig.tight_layout()
    fig.savefig(filename)


def plot_cf(testy, predy, title, filename):  # pragma: no cover
    '''
    Uses sklearn metric to compute a confusion matrix for visualization
    Inputs:
    testy: array/vector with ground-truth labels for test/evaluation set
    predy: array/vector with predicted sample labels from trained model
    title: string title for plot
    filename: string with extension for confusion matrix file
    '''

    cf_matrix = confusion_matrix(testy, predy)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',
                     vmin=0, vmax=100, ax=ax, annot_kws={'fontsize': 40})
    # increase fontsize on colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)

    ax.set_title(title, fontsize=30)
    ax.set_xlabel('\nPredicted Values', fontsize=25)
    ax.set_ylabel('Actual Values ', fontsize=25)

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0(SNM)', '1(other)'], fontsize=25)
    ax.yaxis.set_ticklabels(['0(SNM)', '1(other)'], fontsize=25)
    # Save the visualization of the Confusion Matrix.
    plt.savefig(filename)
