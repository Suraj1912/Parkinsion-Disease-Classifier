import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('pd_speech_features.csv')

target = dataset['class']
features = dataset.drop(['id', 'class'], axis=1)

new_features = SelectKBest(f_classif, k=10).fit(features, target)

new_dataset = pd.DataFrame({'Feature': list(features.columns), 'Scores':new_features.scores_})
new_dataset = new_dataset.sort_values(by='Scores', ascending=False)

new_features = new_features.transform(features)
columns = new_dataset.iloc[:10, 0].values

X = pd.DataFrame(new_features, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size= 0.3)



scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

C = [1.0, 1.5, 2.0, 3.0, 2.5]

kernel = ['linear', 'poly', 'rbf', 'sigmoid']

grid = GridSearchCV(estimator=SVC(), param_grid={'C' : C, 'kernel' : kernel} )
grid.fit(X_train_scaled, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train_scaled, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(X_train_scaled, y_train)

knc = KNeighborsClassifier()
knc.fit(X_train_scaled, y_train)

logr = LogisticRegression()
logr.fit(X_train_scaled, y_train)

rdf = RandomForestClassifier()
rdf.fit(X_train_scaled, y_train)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

pickle.dump(grid, open('svm_model.pkl', 'wb'))

pickle.dump(mlp, open('mlp_model.pkl', 'wb'))

pickle.dump(dtc, open('dtc_model.pkl', 'wb'))

pickle.dump(knc, open('knc_model.pkl', 'wb'))

pickle.dump(logr, open('log_model.pkl', 'wb'))

pickle.dump(rdf, open('rdf_model.pkl', 'wb'))
