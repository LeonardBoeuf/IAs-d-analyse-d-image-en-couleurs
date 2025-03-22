from projet_utils import *

def euclidian_distance(v1,v2):
    return np.sqrt(np.sum((v1 - v2)**2))


def neighbors(X_train, y_label, x_test, k=5):
    list_distances =  []
    for i in range(X_train.shape[0]):
        distance = euclidian_distance(X_train[i], x_test)
        list_distances.append(distance)

    df = pd.DataFrame()
    df["label"] = y_label
    df["distance"] = list_distances
    df = df.sort_values(by="distance")
    return df.iloc[:k,:]

def prediction(dataframe):
    l = [0 for _ in range(10)]
    for i in range(dataframe.shape[0]):
        l[int(dataframe["label"].iloc[i])] += 1
    return np.argmax(l)

def evaluation(X_train,Y_train,X_valid,Y_valid,k=5,verbose=True):
    correct = 0
    erreur = 0

    for image in range(X_valid.shape[0]):
        if verbose and image%10==0:
            print(str(image/X_valid.shape[0]*100)+"%")
        voisins=neighbors(X_train,Y_train,X_valid[image],k)
        if (prediction(voisins) == Y_valid[image]):
            correct += 1
        else:
            erreur += 1
    return correct, erreur, correct / len(X_valid)


def output(X_train,Y_train,X_valid,k=5,verbose=False):
    tab=[]
    for image in range(X_valid.shape[0]):
        if verbose and image%10==0:
            print(str(image/X_valid.shape[0]*100)+"%")
        voisins=neighbors(X_train,Y_train,X_valid[image],k)
        tab.append(prediction(voisins))
    return tab

def graphe_meilleur_k(dict,verbose=True):
    dict["data"]=normalize(dict["data"])
    X_train , X_valid, Y_train, Y_valid =  train_test_split(dict["data"],dict["target"],test_size=0.5,random_state=42)
    tab=[]
    taille_train=500
    taille_valid=50
    r=range(1,100,2)
    for i in r:
        acc=evaluation(X_train[:taille_train],Y_train[:taille_train],X_valid[:taille_valid],Y_valid[:taille_valid],i,False)[2]
        tab.append(acc)
        if verbose:
            print(acc)
    if verbose:
        plt.plot(r,tab)
        plt.show()
    return r[np.argmax(tab)]

def k_plus_proches_voisins(train,valid,k,verbose=False):
    train["data"]=normalize(train["data"])
    valid["data"]=normalize(valid["data"])
    return(output(train["data"],train["target"],valid["data"],k,verbose))



if(__name__=="__main__"):
    with open("dataset_images_train", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    deezplay(dict,0)

    afficherclasse(dict,0)

    #faireTNSE(dict)

    k=graphe_meilleur_k(dict,False)
    print(k)

    with open("data_images_test", 'rb') as fo:
        dict2 = pickle.load(fo, encoding='bytes')
    T=k_plus_proches_voisins(dict,dict2,k,True)
    print(T)
    np.savetxt("images_test_predictions.csv",T)






