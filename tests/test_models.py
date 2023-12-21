# diagnostics
import numpy as np
import pytest
from datetime import datetime, timedelta
# testing models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import tests.test_data as test_data
# hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp
# testing utils
import scripts.utils as utils
# models
from models.LogReg import LogReg
from models.SSML.CoTraining import CoTraining
# testing write
import joblib
import os


@pytest.fixture(scope='module', autouse=True)
def data_init():
    # initialize sample data
    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)
    pytest.timestamps = np.arange(start_date,
                                  start_date + (test_data.timesteps * delta),
                                  delta
                                  ).astype('datetime64[s]').astype('float64')

    pytest.live = np.full((len(pytest.timestamps),), test_data.livetime)
    # 4/5 of the data will be separable
    spectra_sep, labels_sep = make_classification(
                                n_samples=int(0.8*len(pytest.timestamps)),
                                n_features=test_data.energy_bins,
                                n_informative=test_data.energy_bins,
                                n_redundant=0,
                                n_classes=2,
                                class_sep=10.0)
    # 1/5 of the data will be much less separable
    spectra_unsep, labels_unsep = make_classification(
                                    n_samples=int(0.2*len(pytest.timestamps)),
                                    n_features=test_data.energy_bins,
                                    n_informative=test_data.energy_bins,
                                    n_redundant=0,
                                    n_classes=2,
                                    class_sep=0.5)
    pytest.spectra = np.append(spectra_sep, spectra_unsep, axis=0)
    pytest.labels = np.append(labels_sep, labels_unsep, axis=0)


def test_cross_validation():
    # test cross validation for supervised data using LogReg
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0}
    model = LogReg(params=params)
    results = utils.cross_validation(model=model,
                                     X=pytest.spectra,
                                     y=pytest.labels,
                                     params=params,
                                     n_splits=5,
                                     shuffle=False)
    accs = [res['accuracy'] for res in results]
    # the last K-fold will use the unseparable data for testing
    # therefore its accuracy should be less than all other folds
    assert (accs[-1] < accs[:-1]).all()

    # test cross validation for SSML with LabelProp
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
    X, Ux, y, Uy = train_test_split(pytest.spectra,
                                    pytest.labels,
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

    filename = 'test_multiD_pca'
    pcs = utils.pca(X_train, Ux, 5)
    utils.plot_pca(pcs, y_train, np.full_like(Uy, -1), filename, 5)
    os.remove(filename+'.png')

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
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0, 'random_state': 0}
    model = LogReg(max_iter=params['max_iter'],
                   tol=params['tol'],
                   C=params['C'],
                   random_state=params['random_state'])

    assert model.model.max_iter == params['max_iter']
    assert model.model.tol == params['tol']
    assert model.model.C == params['C']
    assert model.random_state == params['random_state']

    X_train, X_test, y_train, y_test = train_test_split(pytest.spectra,
                                                        pytest.labels,
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

    # since the test data used here is synthetic/toy data (i.e. uninteresting),
    # the trained model should be at least better than a 50-50 guess
    # if it was worse, something would be wrong with the ML class
    assert acc > 0.5

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

    # again, the data does not guarantee that an improvement will be found
    # from hyperopt, so long as something is not wrong with the class
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
    # test saving model input parameters
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0,
              'random_state': 0, 'seed': 1}
    model = CoTraining(max_iter=params['max_iter'],
                       tol=params['tol'],
                       C=params['C'],
                       random_state=params['random_state'],
                       seed=params['seed'])

    assert model.model1.max_iter == params['max_iter']
    assert model.model1.tol == params['tol']
    assert model.model1.C == params['C']

    assert model.model2.max_iter == params['max_iter']
    assert model.model2.tol == params['tol']
    assert model.model2.C == params['C']

    assert model.random_state == params['random_state']
    assert model.seed == params['seed']

    X, Ux, y, Uy = train_test_split(pytest.spectra,
                                    pytest.labels,
                                    test_size=0.5,
                                    random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    # normalization
    normalizer = StandardScaler()
    normalizer.fit(X_train)

    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    Ux = normalizer.transform(Ux)

    # default behavior
    model = CoTraining(params=None, random_state=0)
    model.train(X_train, y_train, Ux)

    # testing train and predict methods
    pred, acc, *_ = model.predict(X_test, y_test)

    # since the test data used here is synthetic/toy data (i.e. uninteresting),
    # the trained model should be at least better than a 50-50 guess
    # if it was worse, something would be wrong with the ML class
    assert acc > 0.5

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
                                                1)),
             'seed': 0
             }
    data_dict = {'trainx': X_train,
                 'testx': X_test,
                 'trainy': y_train,
                 'testy': y_test,
                 'Ux': Ux
                 }
    model.optimize(space, data_dict, max_evals=2, verbose=True)

    assert model.best['accuracy'] >= model.worst['accuracy']
    assert model.best['status'] == 'ok'

    # testing model plotting method
    filename = 'test_plot'
    model.plot_cotraining(model1_accs=model.best['model1_acc_history'],
                          model2_accs=model.best['model2_acc_history'],
                          filename=filename)
    os.remove(filename+'.png')

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)
