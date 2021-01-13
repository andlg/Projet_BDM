
#########################  PROJET BIG DATA MINING  ##########################
#                                                                           #
#   Equipe : Alexandre Iborra, Anthony Delage                               #
#   Classification dans un contexte déséquilibré                            #
#   Une application à la fraude bancaire                                    #
#   M2 SISE - Guillaume Metzler                                             #
#                                                                           #
#############################################################################


############## Chargement des librairies ##############

import pandas as pd
import pandas_profiling as pp
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

############## Chargement des données ##############

fraude = pd.read_csv("C:/Users/alexi/Desktop/ProjetML/dataproject.txt",sep=";",decimal=",")


############## Pré-traitement des données  ##############

# Modification des types de certaines variables
fraude['FlAgImpAye'] = fraude['FlAgImpAye'].astype(str) 
fraude['CodeDecision'] = fraude['CodeDecision'].astype(str) 
fraude['IDAvisAutorisAtionCheque'] = fraude['IDAvisAutorisAtionCheque'].astype(str) 

dummy = pd.get_dummies(fraude['CodeDecision'], prefix='Code')

fraude2 = fraude.drop('CodeDecision',axis='columns')
fraude2 = pd.concat([fraude2,dummy],axis=1)

Quanti = ['MontAnt','VerifiAnceCPT1','VerifiAnceCPT2','VerifiAnceCPT3','D2CB','ScoringFP1','ScoringFP2','ScoringFP3','TAuxImpNb_RB','TAuxImpNB_CPM','EcArtNumCheq','NbrMAgAsin3J','DiffDAteTr1','DiffDAteTr2','DiffDAteTr3','CA3TRetMtt','CA3TR','Heure'] 
features = fraude2[Quanti] 
scaler = StandardScaler().fit(features.values) 
features = scaler.transform(features.values) 
fraude2[Quanti] = features



############## Statistiques descriptives ##############

fraude2.head(5)
fraude2.info()

report = pp.ProfileReport(fraude)
report.to_file('profile_report.html')

############## Séparation des données train et test ##############

fraude_train = fraude2.iloc[0:1967225]
fraude_test = fraude2.iloc[1967226:len(fraude.index)]

var = ['MontAnt','VerifiAnceCPT2','D2CB','ScoringFP1','ScoringFP2','ScoringFP3','TAuxImpNB_CPM','EcArtNumCheq','NbrMAgAsin3J','CA3TRetMtt','CA3TR'] 

y_train = fraude_train["FlAgImpAye"]
y_test = fraude_test["FlAgImpAye"]
X_train = fraude_train[var]
X_test = fraude_test[var]


##### Pre-Process #####

rUs = RandomUnderSampler()
X_ru, y_ru = rUs.fit_resample(X_train, y_train)

rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train, y_train)

smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)


############## Implémentation des algos ##############

##### SVM #####

mod_svm =  SVC(kernel='linear')

mod_svm.fit(X_ru, y_ru)
pred_svm = mod_svm.predict(X_test)
confusion_matrix(y_test,pred_svm)
print(classification_report(y_test,pred_svm))

##### Gradient boosting #####

gbbc = BalancedBaggingClassifier()
gbbc.fit(X_train, y_train) 

y_pred_gbbc = gbbc.predict(X_test)
confusion_matrix(y_test,y_pred_gbbc)
print(classification_report(y_test,y_pred_gbbc))


##### Random forest #####

bclf = BalancedRandomForestClassifier()
bclf.fit(X_train, y_train) 

y_pred_rf = bclf.predict(X_test)
confusion_matrix(y_test,y_pred_rf)
print(classification_report(y_test,y_pred_rf))

fraude_test['FlAgImpAye'].value_counts()

############## Comparaison des résultats ##############