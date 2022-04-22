import numpy as np
# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# torch imports
import torch
# shadow imports
import shadow

shadow.utils.set_seed(0)  # set seeds for reproducibility


def model_factory(length=1000, hidden_layer=10000):
    return torch.nn.Sequential(
        torch.nn.Linear(length, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, length),
        torch.nn.ReLU(),
        torch.nn.Linear(length, 2)
    )


def f_nn(params):
    device = torch.device('cpu')  # run on cpu, since model and data are very small
    eaat = shadow.eaat.EAAT(model=model_factory(testx[:,::params['binning']].shape[1], params['hidden_layer']), alpha=params['alpha'], xi=params['xi'], eps=params['eps']).to(device)
    eaat_opt = torch.optim.SGD(eaat.parameters(), lr=params['lr'], momentum=params['momentum'])
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    # avoid float round-off by using DoubleTensor
    xtens = torch.FloatTensor(np.append(trainx, U[:,1:], axis=0)[:,::params['binning']])
    # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
    ytens = torch.LongTensor(np.append(trainy, U[:,0], axis=0))
    #n_epochs = params['n_epochs']
    n_epochs = 100
    xt, yt = torch.Tensor(xtens).to(device), torch.LongTensor(ytens).to(device)
    acc_history = []    # saves history for max accuracy
    eaat.train()
    for epoch in range(n_epochs):
        # Forward/backward pass for training semi-supervised model
        out = eaat(xt)
        loss = xEnt(out, yt) + eaat.get_technique_cost(xt)  # supervised + unsupervised loss
        eaat_opt.zero_grad()
        loss.backward()
        eaat_opt.step()
    
        eaat.eval()
        eaat_pred = torch.max(eaat(torch.FloatTensor(testx.copy()[:,::params['binning']])), 1)[-1]
        acc = shadow.losses.accuracy(eaat_pred, torch.LongTensor(testy.copy())).data.item()
        acc_history.append(acc)
    max_acc = np.max(acc_history[-50:])

    return {'loss': 1-(max_acc/100.0),
            'status': STATUS_OK,
            'model': eaat,
            'params': params,
            'accuracy': (max_acc/100.0)}