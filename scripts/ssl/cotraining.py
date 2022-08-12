import numpy as np
import matplotlib.pyplot as plt
# For hyperopt (parameter optimization)
from scripts.utils import STATUS_OK
# sklearn models
from sklearn import linear_model
# diagnostics
from sklearn.metrics import balanced_accuracy_score

split_frac = 0.5
# labeled training data
idx = np.random.choice(range(trainy.shape[0]),
                        size=int(split_frac * trainy.shape[0]),
                        replace = False)


def f_ct(params):
    slr1 = linear_model.LogisticRegression(random_state=0, max_iter=params['max_iter'], tol=params['tol'], C=params['C'])#, multi_class='multinomial')
    slr2 = linear_model.LogisticRegression(random_state=0, max_iter=params['max_iter'], tol=params['tol'], C=params['C'])#, multi_class='multinomial')

    L_lr1 = trainx[idx].copy()
    L_lr2 = trainx[~idx].copy()
    Ly_lr1 = trainy[idx].copy()
    Ly_lr2 = trainy[~idx].copy()
    # unlabeled cotraining data
    U_lr = U[:,1:].copy()

    model1_accs, model2_accs = np.array([]), np.array([])
    n_samples = params['n_samples']
    rep = False

    while U_lr.shape[0] > 1:
        #print(U_lr.shape[0])
        slr1.fit(L_lr1, Ly_lr1)
        slr2.fit(L_lr2, Ly_lr2)

        # pull u1
        if U_lr.shape[0] < n_samples*2:
            n_samples = int(U_lr.shape[0]/2)
        uidx1 = np.random.choice(range(U_lr.shape[0]), n_samples, replace=rep)
        #u1 = U_lr[uidx1].copy().reshape((1, U_lr[uidx1].shape[0]))
        u1 = U_lr[uidx1].copy()
        U_lr = np.delete(U_lr, uidx1, axis=0)

        # pull u2
        uidx2 = np.random.choice(range(U_lr.shape[0]), n_samples, replace=rep)
        #u2 = U_lr[uidx2].copy().reshape((1, U_lr[uidx2].shape[0]))
        u2 = U_lr[uidx2].copy()
        U_lr = np.delete(U_lr, uidx2, axis=0)

        # predict unlabeled samples
        u1y = slr1.predict(u1)
        u2y = slr2.predict(u2)

        model1_accs = np.append(model1_accs, balanced_accuracy_score(testy, slr1.predict(testx)))
        model2_accs = np.append(model2_accs, balanced_accuracy_score(testy, slr2.predict(testx)))

        # send predictions to cotrained function samples
        L_lr1 = np.append(L_lr1, u2, axis=0)
        L_lr2 = np.append(L_lr2, u1, axis=0)
        Ly_lr1 = np.append(Ly_lr1, u2y, axis=0)
        Ly_lr2 = np.append(Ly_lr2, u1y, axis=0)

    model1_acc = balanced_accuracy_score(testy, slr1.predict(testx))
    model2_acc = balanced_accuracy_score(testy, slr2.predict(testx))
    acc = max(model1_acc, model2_acc)
    return {'loss': 1-acc,
            'status': STATUS_OK,
            'model': slr1,
            'model2': slr2,
            'model1_acc_history': model1_accs,
            'model2_acc_history': model2_accs,
            'params': params,
            'accuracy': acc}


def plot_cotraining():
    plt.plot(np.arange(len(best_ct['model1_acc_history'])), best_ct['model1_acc_history'], label='Model 1')
    plt.plot(np.arange(len(best_ct['model2_acc_history'])), best_ct['model2_acc_history'], label='Model 2')
    plt.legend()
    plt.xlabel('Co-Training Iteration')
    plt.ylabel('Test Accuracy')
    plt.grid()
    plt.savefig('lr-cotraining-learningcurves.png')