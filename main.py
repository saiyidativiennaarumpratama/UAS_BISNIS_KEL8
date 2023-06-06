import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import time

# from proses import preprocessing
from proses.preprocessing import minMax
from proses import model
from proses import implementasi


st.markdown("# UAS KECERDASAN BISNIS")
st.write(
    """
    ##### Kelompok 8 \n
    Rosita Dewi Lutfiyah          200411100002 \n
    Saiyidati Vienna Arum Pratama 200411100018 \n
    Rizki Aji Santoso             20041110017"""
)

st.markdown("# Klasifikasi Data Garam Menggunakan Metode Naive Bayes")
st.text(""" Data yang digunakan untuk klasifikasi memiliki 7 fitur dan 1 target """)

st.markdown("# Pengolahan Data")
selected = option_menu(
    menu_title="Kecerdasan Bisnis",
    options=["Dataset", "Preprocessing", "Modeling", "Implementation"],
    orientation="horizontal",
)

X = pd.read_csv("data/dataset_garam.csv")
y = X['Grade']
if (selected == "Dataset"):
    st.success(
        f"Jumlah Data : {X.shape[0]} Data, dan Jumlah Fitur : {X.shape[1]} Fitur")
    dataframe, keterangan = st.tabs(['Dataset', 'Keterangan'])
    with dataframe:
        st.write(X)

    with keterangan:
        st.text("""
             Column:
             - Kadar air : Kadar air mengacu pada jumlah atau persentase air yang terkandung dalam suatu zat atau lingkungan tertentu
             - Tak larut : Tak larut adalah sifat suatu zat yang tidak dapat larut atau larut dengan sangat sedikit dalam pelarut tertentu pada suhu dan kondisi tertentu.
             - Kalsium : Kalsium adalah unsur kimia dengan simbol Ca dan nomor atom 20. Ini adalah logam alkali tanah yang termasuk dalam kelompok 2 tabel periodik. 
             - Magnesium : Magnesium adalah unsur kimia dengan simbol Mg dan nomor atom 12
             - Sulfat : Sulfat adalah ion negatif (anion) yang terdiri dari satu atom sulfur dan empat atom oksigen
             - NaCl(wb) : NaCl (wb) adalah singkatan dari "Natrium Klorida (basis berat)" atau "Sodium Chloride (wet basis)" dalam bahasa Inggris
             - NaCl(db): NaCl (db) adalah singkatan dari "Natrium Klorida (dry basis)" atau "Sodium Chloride (dry basis)"
             
             Label
             Output Dari Dataset ini yaitu K1 K2 K3 dan K4
           """)

########################################## Preprocessing #####################################################
elif (selected == 'Preprocessing'):
    minMax()


elif selected == 'Modeling':
    st.markdown("# Model Naive Bayes")
    # Menangkap Confusion Matrix dan akurasi yang dikembalikan
    cm_gnb, ac_gnb = model.gnbclassifier()

    st.write("Confusion Matrix:")
    st.write(cm_gnb)

    st.success("Akurasi Naive Bayes Gaussian: " + str(ac_gnb) + "%")

####################### Implementasi ############################


elif selected == 'Implementation':
    Kadar_air = st.number_input('Input nilai Kadar Air')
    Tak_larut = st.number_input('Input nilai Tak Larut')
    Kalsium = st.number_input('Input nilai Kalsium')
    Magnesium = st.number_input('Input nilai Magnesium')
    Sulfat = st.number_input('Input nilai Sulfat')
    NaCl_wb = st.number_input('Input nilai NaCl (wb)')
    NaCl_db = st.number_input('Input nilai NaCl (db)')

    # Melakukan normalisasi pada data input
    data_input = [[Kadar_air, Tak_larut, Kalsium,
                   Magnesium, Sulfat, NaCl_wb, NaCl_db]]

    # Memuat model
    scaler = joblib.load('model/df_scaled(norm).save')
    data_input_scaled = scaler.transform(data_input)

    button = st.button('Prediksi')
    if button:
        grade_prediction = implementasi.nb(data_input_scaled)
        # st.write("Hasil Prediksi: ", grade_prediction)
