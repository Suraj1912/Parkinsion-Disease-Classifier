import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif, chi2

def selectkBest(k, features, target):
    new_features = SelectKBest(f_classif, k=k).fit(features, target)
    st.write('Features with scores')
    new_dataset = pd.DataFrame({'Feature': list(features.columns), 'Scores':new_features.scores_})
    new_dataset = new_dataset.sort_values(by='Scores', ascending=False)
    st.write(new_dataset)
    new_features = new_features.transform(features)
    columns = new_dataset.iloc[:k, 0].values
    st.write(f'Top {k} features are selected')
    X = pd.DataFrame(new_features, columns=columns)
    st.write(X)
    st.write('Shape of new dataset', X.shape)
    return X