from projet_utils import *
import torch as th
import torch.nn.functional as F
import torch.nn as nn

def prediction(f):
    return th.argmax(f, 1)

#3) Fonction qui calcule le taux d'erreur en comparant les y prédits avec les y réels
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]


#4) Création du réseau de neurones. Il étend la classe th.nn.Module de la librairie Pytorch
class Neural_network_multi_classe(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d,k,h1,h2,dropout):
        super(Neural_network_multi_classe, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)

        self.layer1 = th.nn.Linear(32*16*16, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*16*16)

        x = th.sigmoid(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = th.sigmoid(self.layer2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return th.softmax(self.layer3(x),1)

#4) creation d'un réseau de neurones avec deux couches cachées de taille 200 et 100, et k sorties (ici 10 car on veut prédire les valeurs de 0 à 10 des chiffres manuscrits).
if(__name__=="__main__"):
    d = 3072
    k = 10
    #chargement des données
    with open("dataset_images_train", 'rb') as fo:
        train = pickle.load(fo, encoding='bytes')

    X_train=train["data"]
    Y_train=train["target"]

    X_train=normalize(X_train)

    trainloader = th.utils.data.DataLoader(X_train, batch_size=64, shuffle=True)
    valloader = th.utils.data.DataLoader(Y_train, batch_size=64, shuffle=True)





    #Meta paramètres (Im So Meta, Even This Acronym):

    nnet = Neural_network_multi_classe(d,k,200,100,0.2)


    #Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
    device = "cpu"

    nnet = nnet.to(device)
    criterion = th.nn.CrossEntropyLoss()
    images, labels = next(iter(trainloader))
    #images = images.view(images.shape[0], -1)
    images = images.to(device)
    lr = 0.001

    taille_train=20000

    X_train = th.from_numpy(X_train[:taille_train]).float().to(device)
    Y_train = th.from_numpy(Y_train[:taille_train]).long().to(device)

    #optimizer = optim.SGD(nnet.parameters(), lr=lr, momentum = 0.9)
    optimizer = optim.Adam(nnet.parameters())

    nb_epochs = 1000#00
    pbar = tqdm(range(nb_epochs))

    error_test = np.nan

    for epoch in pbar:

        cpt_batch = 0

        for images, labels in trainloader:


            # envoi ds données sur le device
            images = images.to(device)
            labels = labels.to(device)

            # Remise à zéro des gradients
            optimizer.zero_grad()

            f_train = nnet(images)

            loss = criterion(f_train,labels)

            ###Ajout d'une régularization L2
            l2_lambda = 0.001
            l2_reg = 0

            for param in nnet.parameters():
                l2_reg += torch.norm(param)

            loss += l2_lambda * l2_reg


            # Calculs des gradients
            loss.backward()

            # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
            optimizer.step()

            cpt_batch += 1

            y_pred_train = prediction(f_train)
            error_train = error_rate(y_pred_train, labels)


            pbar.set_postfix(iter=epoch, idx_batch = cpt_batch, loss=loss.item(), error_train=error_train.item(), error_test = error_test, l2_reg=l2_reg)


        #test sur l'ensemble de validation à la fin de chaque epoch
        error_avg = 0
        all_count   = 0

        for images, labels in valloader:


            #images = images.view(images.shape[0], -1)

            images = images.to(device)
            labels = labels.to(device)

            f_test = nnet(images)

            # Affichage des 5 premières images avec les prédictions associées
            if(all_count == 0):
                probas = f_test.cpu().detach().numpy()




            y_pred_test = prediction(f_test)
            error_avg += error_rate(y_pred_test, labels)
            all_count += 1

        print("labels")
        print(labels)

        print("y_pred_test")
        print(y_pred_test)

        error_test = (error_avg/all_count).item()

        pbar.set_postfix(iter=epoch, idx_batch=cpt_batch, loss=loss.item(), error_train=error_train, error_test = error_test)