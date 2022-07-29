import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from hyperopt import Trials, tpe, fmin
from functools import partial
# diagnostics
from sklearn.metrics import confusion_matrix


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
    fmin_objective = partial(model, data_dict=data_dict, device=None)

    # run hyperopt
    optimizer = fmin(fmin_objective, 
                     space, 
                     algo=tpe.suggest,
                     max_evals=max_evals,
                     trials=trials)

    # of all trials, find best and worst loss/accuracy from optimization
    best = trials.results[np.argmin([r['loss'] for r in 
        trials.results])]
    worst = trials.results[np.argmax([r['loss'] for r in 
        trials.results])]
    
    if verbose:
        print('best accuracy:', 1-best['loss'])
        print('best params:', best['params'])
        print('worst accuracy:', 1-worst['loss'])
        print('worst params:', worst['params'])
    
    return best, worst


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

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0(SNM)','1(other)'])
    ax.yaxis.set_ticklabels(['0(SNM)','1(other)'])
    ## Save the visualization of the Confusion Matrix.
    plt.savefig(filename)
