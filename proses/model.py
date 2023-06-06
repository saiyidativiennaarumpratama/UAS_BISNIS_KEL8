import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st


X_mod = pd.read_csv("data/dataset_garam.csv")
X = X_mod.drop(columns=['Grade'])
Y = X_mod['Grade']

def gnbclassifier():
    # preprocessing min max scaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(X)
    #memasukan fitur 
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(data_scaled, columns=features_names)
    # scaled_features = "df_scaled(norm).save"
    X_training, X_test, Y_training_label, Y_test_label = train_test_split(scaled_features,Y,test_size=0.1, random_state=42, shuffle=True)
    
    gnbclassifier = GaussianNB()
    # data training ntuk pembelajarannya
    # test
    param={'var_smoothing': np.logspace(0,-9, num=1000)}

    nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param)
    nbModel_grid.fit(X_training, Y_training_label)
    gnbclassifier = (nbModel_grid.best_estimator_)

    Y_pred = gnbclassifier.predict(X_test)

    # Menghitung confusion matrix dari data testing terhadap data prediksi
    cm_gnb = confusion_matrix(Y_test_label, Y_pred)
    # return cm_gnb
    # st.write("Confusion Matrix:")
    # st.write(cm_gnb)


    ac_gnb = round (accuracy_score(Y_test_label, Y_pred)*100)
    # st.success("Akurasi Gaussian Naive Bayes (in %): ", ac_gnb)
    # st.success("Akurasi Gaussian Naive Bayes (in %): " + str(ac_gnb))

    # joblib.dump(nb_np, 'model/nb_model.sav')
    # filename = 'modelNB.pkl'
    joblib.dump(gnbclassifier, 'model/modelNB.pkl')
    return cm_gnb, ac_gnb

gnbclassifier()