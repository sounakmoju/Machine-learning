from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import random
def accuracy_score(y_true, y_pred):
    
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy



def train_test_split(X,y,test_size):
    
    if isinstance(test_size,float):
        test_size=round(test_size*len(y))
        train_size=len((y))-test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test
    
class stump():
    def __init__(self):
        self.pole=1
        self.feature_index=None
        self.value=None
        self.alpha=None 
    


class Adaboost():

    def __init__(self, n_clf=5):
        self.n_clf=n_clf
    def fit(self,X,y):
        n_samples,n_features=np.shape(X)
        w=np.full(n_samples,(1/n_samples))

        self.clfs=[]
        for _ in range(self.n_clf):
            clf=stump()
            min_error=999999

            for i in range(0,n_features):
                features=np.expand_dims(X[:,i],axis=1)
                splitvalues=np.unique(features)
                #print(i)

                for j in splitvalues:
                    p=1
                    prediction=np.ones(np.shape(y))
                    prediction[X[:,i]<j]=-1
                    error=sum(w[y!=prediction])
                    #print(prediction)

                    if error>0.5:
                        error=1-error
                        p=-1
                    if error<min_error:
                        clf.pole=p
                        clf.feature_index=i
                        clf.value=j
                        min_error=error
                        #print(clf.value)
                    #print(clf.feature_index)
                    #print(error)
                        
            clf.alpha=0.5*math.log((1-min_error)/(min_error+1e-10))
            #print(clf.alpha)
            predictions=np.ones(np.shape(y))

            neg_idx=(clf.pole*X[:,clf.feature_index]<clf.pole*clf.value)
            predictions[neg_idx]=-1
            w *= np.exp(-clf.alpha * y * predictions)
            w/=np.sum(w)
            self.clfs.append(clf)
            #print(clf.feature_index)
            

    def predict(self,X):

        n_samples=np.shape(X)[0]
        y_pre=np.zeros((n_samples,1))
        for clf in self.clfs:
            predictions=np.ones(np.shape(y_pre))
            negative_idx=(clf.pole*X[:,clf.feature_index]< clf.pole*clf.value)
            predictions[negative_idx]=-1
            y_pre+=clf.alpha*predictions

        y_pre=np.sign(y_pre).flatten()
        print(y_pre)

        return y_pre
def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
    print(X_train)
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(y_pred)
    print ("Accuracy:", accuracy)
    plt.plot(y_test, y_pred)
    plt.show()
    #print(X_test)


if __name__ == "__main__":
    main()

        

    
    


       
        
            
                    

