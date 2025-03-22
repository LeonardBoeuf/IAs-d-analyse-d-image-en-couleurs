import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def nomclasse(c):
        if(c==0):
            return("plané")
        elif(c==1):
            return("BAH ALORS GROSSE MERDE ON A PAS LE PERIMS")
        elif(c==2):
            return("zozio")
        elif(c==3):
            return("SHAW")
        elif(c==4):
            return("ser")
        elif(c==5):
            return("chien")
        elif(c==6):
            return("guenouille")
        elif(c==7):
            return("cheval")
        elif(c==8):
            return("mouton")
        elif(c==9):
            return("TUT TUT")
        else:
            return(c)


def afficher(dict, a):
        c=dict["target"][a]
        print(nomclasse(c))

def display(dict, a,affich=True,sho=True):
    image=dict["data"][a].reshape(3,32,32)
    if affich:
        affichere(dict,a)
    image=np.transpose(image,(1,2,0))
    plt.imshow(image)
    if sho:
        plt.show()

def afficherclasse(dict,c):
    i=0
    nb=0
    while(nb<10 and i<len(dict["target"])):
        if (dict["target"][i]==c):
            display(dict,i,False)
            nb=nb+1
        i=i+1

def normalize(X):
    return preprocessing.normalize(X)

def faireTNSE(dict):
    data_tnse=TSNE(n_components=2,learning_rate="auto",init="random",perplexity=3).fit_transform(dict["data"][0:3000])

    plt.figure(figsize=(8,6))
    scatt=plt.scatter(data_tnse[:,0],data_tnse[:,1],c=dict["target"][0:3000])
    plt.colorbar(scatt)
    plt.show()

def afficherclassesupposee (dict,c):
    i=0
    nb=0
    tab =np.loadtxt("images_test_predictions.csv")
    while(nb<10 and i<len(dict["target"])):
        if (tab[i]==c):
            display(dict,i,False)
            nb=nb+1
        i=i+1

if(__name__=="__main__"):
    with open("dataset_images_train", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    """display(dict,0)

    afficherclasse(dict,8)"""
    c=8
    print(nomclasse(c))
    afficherclassesupposee(dict,c)
    #faireTNSE(dict)







