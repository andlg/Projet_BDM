############ Projet Big Data ############

import pandas as pd
import pandas_profiling as pp
import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix

## Chargement des données

data = pd.read_csv("C:/Users/alexi/Desktop/ProjetML/dataproject.txt",sep=";",decimal=",")
data.head(5)
data.info()

data['FlAgImpAye'] = data['FlAgImpAye'].astype(str) 
data['CodeDecision'] = data['CodeDecision'].astype(str) 
data['IDAvisAutorisAtionCheque'] = data['IDAvisAutorisAtionCheque'].astype(str) 

Quanti = ['MontAnt','VerifiAnceCPT1','VerifiAnceCPT2','VerifiAnceCPT3','D2CB','ScoringFP1','ScoringFP2','ScoringFP3','TAuxImpNb_RB','TAuxImpNB_CPM','EcArtNumCheq','NbrMAgAsin3J','DiffDAteTr1','DiffDAteTr2','DiffDAteTr3','CA3TRetMtt','CA3TR','Heure'] 
Quali = ['ZIBZIN','IDAvisAutorisAtionCheque','DAteTrAnsAction','CodeDecision'] 

Y = data["FlAgImpAye"]
X = data.iloc[:,0:22]

X_quanti = X[Quanti]
X_quali = X[Quali]

X_quanti.head()


## Statistiques descriptives sur les données
report = pp.ProfileReport(data)
report.to_file('profile_report.html')


## Selection de variable
# Méthode de filtre
select = SelectKBest(chi2,k=5)
select.fit_transform(X_quanti,Y)

## Méthodes d'échantillonage
data_train = data.iloc[0:1967225]
data_test = data.iloc[1967226:len(data.index)]

data_train['FlAgImpAye'].value_counts()
data_test['FlAgImpAye'].value_counts()

Quanti_bis = ['MontAnt','VerifiAnceCPT2','D2CB','ScoringFP1','ScoringFP2','ScoringFP3','TAuxImpNB_CPM','EcArtNumCheq','NbrMAgAsin3J','CA3TRetMtt','CA3TR'] 
Quali_bis = ['ZIBZIN','IDAvisAutorisAtionCheque','DAteTrAnsAction','CodeDecision'] 

Y_train = data_train["FlAgImpAye"]
Y_test = data_test["FlAgImpAye"]
X_test = data_test[Quanti_bis]
X_train = data_train[Quanti_bis]

## Algoritmes supervisés - paramétrage


Classifier = SGDClassifier(max_iter=10**7//len(X_train)+1) #Model instance, empirically result for max_iter with 10**6
params = {'alpha':[ 1e-3, 1e-4, 1e-5, 1e-6 ], 'penalty':['l1', 'l2'], 'loss':['hinge','log']} #Paramètres à tester
clf = GridSearchCV(Classifier, param_grid=params, cv=5, n_jobs=6)
clf.fit(X_train, Y_train)
#predictions on train data
y_predict=clf.predict(X_test)
 
## Représentation des résultats

print(classification_report(Y_test, y_predict))
confusion_matrix(Y_test, y_predict)

## Séparation train et test
