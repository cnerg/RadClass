import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from hyperopt import Trials, tpe, fmin
from functools import partial
# diagnostics
from sklearn.metrics import confusion_matrix
# pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
        print('best accuracy:', 1-best['loss'])
        print('best params:', best['params'])
        print('worst accuracy:', 1-worst['loss'])
        print('worst params:', worst['params'])

    return best, worst


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
