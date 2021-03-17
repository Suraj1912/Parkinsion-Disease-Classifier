import streamlit as st
import pandas as pd
import numpy as np
import pickle

from features_selection import selectkBest

scaler = pickle.load(open('scaler.pkl', 'rb'))
svc = pickle.load(open('svm_model.pkl', 'rb'))
mlp = pickle.load(open('mlp_model.pkl', 'rb'))
dtc = pickle.load(open('dtc_model.pkl', 'rb'))
knc = pickle.load(open('knc_model.pkl', 'rb'))
log = pickle.load(open('log_model.pkl', 'rb'))
rdf = pickle.load(open('rdf_model.pkl', 'rb'))

def main():
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Parkinson Disease Classifier</h2>
    </div><br>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    dataset = pd.read_csv('pd_speech_features.csv')

    # activity = st.sidebar.selectbox('Select Activity', ['Feature Selection', 'Model Classifier'])

    features, label = divide(dataset)
    
    # if activity == 'Feature Selection':
        
    with st.beta_expander('See Features with their scores'):
        selected_features = selectkBest(10, features, label)

    feat1 = st.slider('mean_MFCC_2nd_coef', min_value=-0.0, max_value=20.0, step=0.01)

    feat2 = st.slider('tqwt_minValue_dec_12', min_value=float(selected_features.iloc[:, 1].min()), max_value=float(selected_features.iloc[:, 1].max()), step=0.01)

    feat3 = st.slider('tqwt_stdValue_dec_12', min_value=float(selected_features.iloc[:, 2].min()), max_value=float(selected_features.iloc[:, 2].max()), step=0.01)

    feat4 = st.slider('tqwt_maxValue_dec_12', min_value=float(selected_features.iloc[:, 3].min()), max_value=float(selected_features.iloc[:, 3].max()), step=0.01)

    feat5 = st.slider('tqwt_stdValue_dec_11', min_value=float(selected_features.iloc[:, 4].min()), max_value=float(selected_features.iloc[:, 4].max()), step=0.01)

    feat6 = st.slider('tqwt_entropy_log_dec_12', min_value=float(selected_features.iloc[:, 5].min()), max_value=float(selected_features.iloc[:, 5].max()), step=0.01)

    feat7 = st.slider('tqwt_maxValue_dec_11', min_value=float(selected_features.iloc[:, 6].min()), max_value=float(selected_features.iloc[:, 6].max()), step=0.01)

    feat8 = st.slider('tqwt_minValue_dec_11', min_value=float(selected_features.iloc[:, 7].min()), max_value=float(selected_features.iloc[:, 7].max()), step=0.01)

    feat9 = st.slider('tqwt_minValue_dec_13', min_value=float(selected_features.iloc[:, 8].min()), max_value=float(selected_features.iloc[:, 8].max()), step=0.01)

    feat10 = st.slider('std_9th_delta_delta', min_value=float(selected_features.iloc[:, 9].min()), max_value=float(selected_features.iloc[:, 9].max()), step=0.01)

    algo = st.sidebar.selectbox('Select Algorithm', ['Logistic regression', 'SVM (best among all)', 'Multi Layer Perceptron', 'Decision Tree Classifier', 'K-Nearest Neighbors', 'Random Forest Classifier'])

    inputs = [[feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10]]

    X_scaled = scaler.transform(inputs)

    if st.button('Apply', 'apply'):

        if algo == 'Logistic regression':
            classify(log.predict(X_scaled))

        if algo == 'SVM (best among all)':
            classify(svc.predict(X_scaled))

        if algo == 'Multi Layer Perceptron':
            classify(mlp.predict(X_scaled))

        if algo == 'Decision Tree Classifier':
            classify(dtc.predict(X_scaled))

        if algo == 'K-Nearest Neighbors':
            classify(knc.predict(X_scaled))

        if algo == 'Random Forest Classifier':
            classify(rdf.predict(X_scaled))

        



def divide(dataset):
    features = dataset.drop(['id', 'class'], axis=1)
    target = dataset['class']

    return features, target

def classify(value):

    if value == 0:
        st.success('Patient have no parkinson disease')
    
    if value == 1:
        st.error('patient is suffering from Parkinson Disease')
        

if __name__ == "__main__":
    main()