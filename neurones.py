from projet_utils import *
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
import torch.cuda

def prediction(f):
    return th.argmax(f, 1)

def error_rate(Y_pred,y):
    return ((Y_pred != y).sum().float())/Y_pred.size()[0]

class Network_Couches_Multiples(th.nn.Module):
    def __init__(self,l,dropout=0.2):
        super(Network_Couches_Multiples, self).__init__()

        self.layers = []
        for i in range(len(l)-1):
            self.layers.append(th.nn.Linear(l[i], l[i+1]))
            self.layers[i].reset_parameters()
        self.layers = th.nn.Sequential(*self.layers)

        self.dropout = dropout


    def forward(self, x):
        phi = F.sigmoid(self.layers[0](x))
        for i in range(1,len(self.layers)):
            phi = F.sigmoid(self.layers[i](phi))
            phi = F.dropout(phi, p=self.dropout, training=self.training)
        return F.softmax(phi,1)

if(__name__=="__main__"):
    d=3072
    k=10
    neurones=[d,50,50,50,k]
    model = Network_Couches_Multiples(neurones)

    #8) Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
    device = "cpu"
    model = model.to(device)

    #chargement des données
    with open("dataset_images_train", 'rb') as fo:
        train = pickle.load(fo, encoding='bytes')

    X_train=train["data"]
    Y_train=train["target"]

    X_train=normalize(X_train)


    taille_train=20000

    X_train = th.from_numpy(X_train[:taille_train]).float().to(device)
    Y_train = th.from_numpy(Y_train[:taille_train]).long().to(device)

    #Meta paramètres (Im So Meta, Even This Acronym):
    criterion = th.nn.CrossEntropyLoss()
    eta= 0.001
    optimizer = optim.Adam(model.parameters(), lr=eta)
    epochs_mult=6
    nb_epochs = int(epochs_mult*5000)#should be 100 000
    pbar = tqdm(range(nb_epochs))

    #exectution :
    for i in pbar:
        # Remise à zéro des gradients
        optimizer.zero_grad()

        f_train = model(X_train)
        loss = criterion(f_train,Y_train)
        # Calculs des gradients
        loss.backward()

        # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
        optimizer.step()
        if (i/nb_epochs*100%10 == 0):
            y_pred_train = prediction(f_train)

            error_train = error_rate(y_pred_train,Y_train)
            loss = criterion(f_train,Y_train)
            print(error_train)


    with open("data_images_test", 'rb') as fo:
        valid = pickle.load(fo, encoding='bytes')
    data=normalize(valid["data"])
    data=th.from_numpy(data).float().to(device)
    f_test = model(data)
    T=prediction(f_test)
    print(T)
    np.savetxt("images_test_predictions.csv",T)
    print("neurones_multi_couches\nneurones = "+str(neurones)+"\ntaille_train =" +str(taille_train)+"\neta = "+str(eta)+"\nOptimizer=Adam\nnbepochs = "+str(nb_epochs))
