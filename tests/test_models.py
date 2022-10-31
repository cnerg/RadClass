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
    assert np.count_nonzero(pred == y_test) > 0

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
