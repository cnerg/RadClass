import numpy as np
# For hyperopt (parameter optimization)
from hyperopt import STATUS_OK
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# shadow imports
import shadow.eaat
import shadow.losses
import shadow.utils
from shadow.utils import set_seed

set_seed(0)
device = torch.device('cpu')  # run on cpu, since model and data are very small

class Net(nn.Module):
    def __init__(self, layer1=32, layer2=64, layer3=128, kernel=3, drop_rate=0.1, length=1000):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, layer1, kernel, 1)
        self.conv2 = nn.Conv1d(layer1, layer2, kernel, 1)
        self.dropout = nn.Dropout2d(drop_rate)
        self.fc1 = nn.Linear(int(layer2*(length-2*(kernel-1))/2), layer3)
        #self.fc1 = nn.Linear(31744, 128)
        self.fc2 = nn.Linear(layer3, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, trainD, labels):
        self.labels = labels
        self.trainD = trainD

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.trainD[idx]
        # no need to bother with labels, unpacking both anyways
        #sample = {"Spectrum": data, "Class": label}
        #return sample
        return data, label

def eval(eaat, binning):
    eaat.eval()
    y_pred, y_true = [], []
    for i, (data, targets) in enumerate(zip(torch.FloatTensor(testx.copy()[:,::binning]), torch.LongTensor(testy.copy()))):
        x = data.reshape((1, 1, data.shape[0])).to(device)
        y = targets.reshape((1,)).to(device)
        out = eaat(x)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(torch.argmax(out, 1).detach().cpu().tolist())
    test_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    #print('test accuracy: {}'.format(test_acc))
    return test_acc

def f_eaat(params):
    #print(params)
    # avoid float round-off by using DoubleTensor
    xtens = torch.FloatTensor(np.append(trainx, U[:,1:], axis=0))[:,::params['binning']]
    # xtens[xtens == 0.0] = torch.unique(xtens)[1]/1e10
    ytens = torch.LongTensor(np.append(trainy, U[:,0], axis=0))
    
    #print(xtens.shape)
    model = Net(layer1=params['layer1'], layer2=2*params['layer1'], layer3=3*params['layer1'], kernel=params['kernel'], drop_rate=params['drop_rate'], length=xtens.shape[1])
    eaat = shadow.eaat.EAAT(model=model, alpha=params['alpha'], xi=params['xi'], eps=params['eps'])
    optimizer = optim.SGD(eaat.parameters(), lr=params['lr'], momentum=params['momentum'])

    # define data set object
    MINOS_train = SpectralDataset(xtens, ytens)

    # create DataLoader object of DataSet object
    DL_DS = torch.utils.data.DataLoader(MINOS_train, batch_size=params['batch_size'], shuffle=True)

    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1)

    n_epochs = 50
    eaat.to(device)
    losscurve = []
    evalcurve = []
    for epoch in range(n_epochs):
        eaat.train()
        lossavg = []
        for i, (data, targets) in enumerate(DL_DS):
            x = data.reshape((data.shape[0], 1, data.shape[1])).to(device)
            y = targets.to(device)
            optimizer.zero_grad()
            out = eaat(x)
            loss = xEnt(out, y) + eaat.get_technique_cost(x)
            loss.backward()
            optimizer.step()
            lossavg.append(loss.item())
        losscurve.append(np.nanmedian(lossavg))
        evalcurve.append(eval(eaat, params['binning']))
    
    max_acc = np.max(evalcurve[-25:])

    return {'loss': 1-(max_acc/100.0),
            'status': STATUS_OK,
            'model': eaat,
            'params': params,
            'accuracy': (max_acc/100.0)}