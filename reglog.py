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

class Reg_log_multi(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d,k):
        super(Reg_log_multi, self).__init__()

        self.layer = th.nn.Linear(d,k)
        self.layer.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        out = self.layer(x)
        return F.softmax(out,1)


if(__name__=="__main__"):
    """
    #definition de la graine aléatoire
    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)
    """
    #Initialisation du modele
    model = Reg_log_multi(3072,10)

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


    #Meta paramètres:
    eta = 0.000065#best till now :0.00007
    criterion = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=eta)
    epochs_mult=1
    nb_epochs = epochs_mult*100000#should be 100 000
    pbar = tqdm(range(nb_epochs))

    #exectution :
    for i in pbar:
        optimizer.zero_grad()

        f_train = model(X_train)
        loss = criterion(f_train,Y_train)
        loss.backward()

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
    print("reglog\ntaille_train = "+str(taille_train)+"\neta = "+str(eta)+"\nOptimizer=Adam\nnbepochs = "+str(nb_epochs))





