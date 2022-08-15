# diagnostics
import numpy as np
from datetime import datetime, timedelta
# testing models
from sklearn.model_selection import train_test_split
import tests.test_data as test_data
# hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp
# models
from models.LogReg import LogReg
from models.SSML.CoTraining import CoTraining
# testing write
import joblib
import os

# initialize sample data
start_date = datetime(2019, 2, 2)
delta = timedelta(seconds=1)
timestamps = np.arange(start_date,
                       start_date + (test_data.timesteps * delta),
                       delta).astype('datetime64[s]').astype('float64')

live = np.full((len(timestamps),), test_data.livetime)
sample_val = 1.0
spectra = np.full((len(timestamps), test_data.energy_bins),
                  np.full((1, test_data.energy_bins), sample_val))
# setting up for rejected null hypothesis
rejected_H0_time = np.random.choice(spectra.shape[0],
                                    test_data.timesteps//2,
                                    replace=False)
spectra[rejected_H0_time] = 100.0

labels = np.full((spectra.shape[0],), 0)
labels[rejected_H0_time] = 1


def test_LogReg():
    X_train, X_test, y_train, y_test = train_test_split(spectra,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=0)

    # default behavior
    model = LogReg(params=None, random_state=0)
    model.train(X_train, y_train)

    # testing train and predict methods
    pred, acc = model.predict(X_test, y_test)

    assert acc > 0.7
    np.testing.assert_equal(pred, y_test)

    # testing hyperopt optimize methods
    space = {'max_iter': scope.int(hp.quniform('max_iter',
                                               10,
                                               10000,
                                               10)),
             'tol': hp.loguniform('tol', 1e-5, 1e-1),
             'C': hp.uniform('C', 0.001, 1000.0)
             }
    data_dict = {'trainx': X_train,
                 'testx': X_test,
                 'trainy': y_train,
                 'testy': y_test
                 }
    model.optimize(space, data_dict, max_evals=10, verbose=True)

    assert model.best['accuracy'] >= model.worst['accuracy']
    assert model.best['status'] == 'ok'

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)


def test_CoTraining():
    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
                                    test_size=0.5,
                                    random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    # default behavior
    model = CoTraining(params=None, random_state=0)
    model.train(X_train, y_train, Ux)

    # testing train and predict methods
    pred, acc, *_ = model.predict(X_test, y_test)

    assert acc > 0.7
    np.testing.assert_equal(pred, y_test)

    # testing hyperopt optimize methods
    space = {'max_iter': scope.int(hp.quniform('max_iter',
                                   10,
                                   10000,
                                   10)),
             'tol': hp.loguniform('tol', 1e-5, 1e-3),
             'C': hp.uniform('C', 1.0, 1000.0),
             'n_samples': scope.int(hp.quniform('n_samples',
                                    1,
                                    20,
                                    1))
             }
    data_dict = {'trainx': X_train,
                 'testx': X_test,
                 'trainy': y_train,
                 'testy': y_test,
                 'Ux': Ux
                 }
    model.optimize(space, data_dict, max_evals=10, verbose=True)

    assert model.best['accuracy'] >= model.worst['accuracy']
    assert model.best['status'] == 'ok'

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)
