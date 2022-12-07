# diagnostics
import numpy as np
import pytest
from datetime import datetime, timedelta
# testing models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tests.test_data as test_data
# hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp
# testing utils
import scripts.utils as utils
# models
from models.LogReg import LogReg
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


def test_cross_validation():
    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
                                    test_size=0.5,
                                    random_state=0)
    Uy = np.full_like(Uy, -1)

    # test cross validation for supervised data using LogReg
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0}
    model = LogReg(params=params)
    max_acc_model = utils.cross_validation(model=model,
                                           X=X,
                                           y=y,
                                           params=params)
    assert max_acc_model['accuracy'] >= 0.5

    # test cross validation for supervised data and StratifiedKFold with LogReg
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0}
    model = LogReg(params=params)
    max_acc_model = utils.cross_validation(model=model,
                                           X=X,
                                           y=y,
                                           params=params,
                                           stratified=True)
    assert max_acc_model['accuracy'] >= 0.5

    # # test cross validation for SSML with LabelProp
    # params = {'gamma': 10, 'n_neighbors': 15, 'max_iter': 2022, 'tol': 0.5}
    # model = LabelProp(params=params)
    # max_acc_model = utils.cross_validation(model=model,
    #                                        X=np.append(X, Ux, axis=0),
    #                                        y=np.append(y, Uy, axis=0),
    #                                        params=params,
    #                                        stratified=True)
    # assert max_acc_model['accuracy'] >= 0.5

def test_pca():
    # unlabeled data split
    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
                                    test_size=0.5,
                                    random_state=0)
    Uy = np.full_like(Uy, -1)
    # data split for data visualization
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    filename = 'test_pca'
    pcs = utils.pca(X_train, Ux, 2)
    utils.plot_pca(pcs, y_train, np.full_like(Uy, -1), filename, 2)
    os.remove(filename+'.png')

    # filename = 'test_multiD_pca'
    # utils.multiD_pca(X_train, y_train, Ux, np.full_like(Uy, -1), filename, n=5)
    # os.remove(filename+'.png')

    # normalization
    normalizer = StandardScaler()
    normalizer.fit(X_train)

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)

    # default behavior
    model = LogReg(params=None, random_state=0)
    model.train(X_train, y_train)

    # testing train and predict methods
    pred, acc = model.predict(X_test, y_test)

    filename = 'test_cf'
    utils.plot_cf(y_test, pred, title=filename, filename=filename)
    os.remove(filename+'.png')


def test_LogReg():
    # test saving model input parameters
    # missing 'C' key
    params = {'max_iter': 2022, 'tol': 0.5}
    with pytest.raises(ValueError):
        LogReg(params=params)
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0}
    model = LogReg(params=params)

    assert model.model.max_iter == params['max_iter']
    assert model.model.tol == params['tol']
    assert model.model.C == params['C']

    X_train, X_test, y_train, y_test = train_test_split(spectra,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=0)

    # normalization
    normalizer = StandardScaler()
    normalizer.fit(X_train)

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)

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
    model.optimize(space, data_dict, max_evals=2, verbose=True)

    assert model.best['accuracy'] >= model.worst['accuracy']
    assert model.best['status'] == 'ok'

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)