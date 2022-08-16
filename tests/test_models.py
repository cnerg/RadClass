# diagnostics
import numpy as np
from datetime import datetime, timedelta
# testing models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tests.test_data as test_data
# hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp
# models
from models.LogReg import LogReg
from models.SSML.CoTraining import CoTraining
from models.SSML.LabelProp import LabelProp
from models.SSML.ShadowNN import ShadowNN
from models.SSML.ShadowCNN import ShadowCNN
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
    # test saving model input parameters
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


def test_CoTraining():
    # test saving model input parameters
    params = {'max_iter': 2022, 'tol': 0.5, 'C': 5.0}
    model = CoTraining(params=params)

    assert model.model1.max_iter == params['max_iter']
    assert model.model1.tol == params['tol']
    assert model.model1.C == params['C']

    assert model.model2.max_iter == params['max_iter']
    assert model.model2.tol == params['tol']
    assert model.model2.C == params['C']

    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
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


def test_LabelProp():
    # test saving model input parameters
    params = {'gamma': 10, 'n_neighbors': 15, 'max_iter': 2022, 'tol': 0.5}
    model = LabelProp(params=params)

    assert model.model.gamma == params['gamma']
    assert model.model.n_neighbors == params['n_neighbors']
    assert model.model.max_iter == params['max_iter']
    assert model.model.tol == params['tol']

    # there should be no normalization on LabelProp data
    # since it depends on the distances between samples
    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
                                    test_size=0.5,
                                    random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    # default behavior
    model = LabelProp(params=None, random_state=0)
    model.train(X_train, y_train, Ux)

    # testing train and predict methods
    pred, acc = model.predict(X_test, y_test)

    # the default n_neighbors(=7) from sklearn is too large
    # for the size of this dataset
    # therefore the accuracy is expected to be poor
    # a better value for this dataset would be n_neighbors=2
    # (tested when specifying params in LabelProp.__init__)
    assert acc >= 0.5
    # uninteresting test if LabelProp predicts all one class
    # TODO: make the default params test meaningful
    assert np.count_nonzero(pred == y_test) > 0

    # testing hyperopt optimize methods
    space = {'max_iter': scope.int(hp.quniform('max_iter',
                                               10,
                                               10000,
                                               10)),
             'tol': hp.loguniform('tol', 1e-6, 1e-4),
             'gamma': hp.uniform('gamma', 1, 50),
             'n_neighbors': scope.int(hp.quniform('n_neighbors',
                                                  1,
                                                  X_train.shape[0],
                                                  1))
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

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)


def test_ShadowNN():
    # check default parameter settings
    model = ShadowNN()
    assert model.params == {'binning': 1}
    assert model.eaat is not None
    assert model.eaat_opt is not None
    assert model.xEnt is not None
    assert model.input_length == 1000

    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
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

    params = {'hidden_layer': 10,
              'alpha': 0.1,
              'xi': 1e-3,
              'eps': 1.0,
              'lr': 0.1,
              'momentum': 0.9,
              'binning': 20}
    # default behavior
    model = ShadowNN(params=params, random_state=0)
    acc_history = model.train(X_train, y_train, Ux, X_test, y_test)

    # testing train and predict methods
    pred, acc = model.predict(X_test, y_test)

    # test for agreement between training and testing
    # (since the same data is used for diagnostics in this test)
    assert acc_history[-1] == acc

    # Shadow/PyTorch reports accuracies as percentages
    # rather than decimals
    # uninteresting test if Shadow predicts all one class
    # TODO: make the default params test meaningful
    # NOTE: .numpy() needed because model.predict() returns a tensor
    assert np.count_nonzero(pred.numpy() == y_test) > 0

    # testing hyperopt optimize methods
    space = {'hidden_layer': 10,
             'alpha': 0.1,
             'xi': 1e-3,
             'eps': 1.0,
             'lr': 0.1,
             'momentum': 0.9,
             'binning': scope.int(hp.quniform('binning',
                                              10,
                                              20,
                                              1))
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

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)


def test_ShadowCNN():
    # check default parameter settings
    model = ShadowCNN()
    assert model.params == {'binning': 1, 'batch_size': 1}
    assert model.model is not None
    assert model.eaat is not None
    assert model.optimizer is not None

    X, Ux, y, Uy = train_test_split(spectra,
                                    labels,
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

    params = {'layer1': 2,
              'kernel': 3,
              'alpha': 0.1,
              'xi': 1e-3,
              'eps': 1.0,
              'lr': 0.1,
              'momentum': 0.9,
              'binning': 20,
              'batch_size': 4,
              'drop_rate': 0.1}

    # default behavior
    model = ShadowCNN(params=params, random_state=0)
    losscurve, evalcurve = model.train(X_train, y_train, Ux, X_test, y_test)

    # testing train and predict methods
    pred, acc = model.predict(X_test, y_test)

    # test for agreement between training and testing
    # (since the same data is used for diagnostics in this test)
    assert evalcurve[-1] == acc

    # Shadow/PyTorch reports accuracies as percentages
    # rather than decimals
    # uninteresting test if Shadow predicts all one class
    # TODO: make the default params test meaningful
    assert np.count_nonzero(pred == y_test) > 0

    # testing hyperopt optimize methods
    space = params
    space['binning'] = scope.int(hp.quniform('binning',
                                 10,
                                 20,
                                 1))
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
    model.plot_training(losscurve=model.best['losscurve'],
                        evalcurve=model.best['evalcurve'],
                        filename=filename)
    os.remove(filename+'.png')

    # testing model write to file method
    filename = 'test_LogReg'
    ext = '.joblib'
    model.save(filename)
    model_file = joblib.load(filename+ext)
    assert model_file.best['params'] == model.best['params']

    os.remove(filename+ext)
