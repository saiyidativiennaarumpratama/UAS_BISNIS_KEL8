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
             - Battery power: Berapa Banyak Power Dari Baterai
             - Blue: Apakah Batrey nya memiliki Bluetooth atau TIDAK
             - Dual_Sim: Apakah Mendukung Dual SIM atau TIDAK
             - fc: Ukuran Pixel Dari Kamera Depan
             - four_g: Apakah Sudah support jaringan 4G atau TIDAK
             - int_memory: Internal Memory berapa GB
             - mobile_wt: Berat Handphone
             - pc: Ukuran Pixel Dari Kamera Belakang/Primary
             - px_height: Pixel Resolution Height
             - px_width: Pixel Resolution Width
             - ram: Ukuran RAM
             - sc_h: Screen Height of mobile in cm
             - sc_w: Screen Width of mobile in cm
             - three_g: Apakah Jaringan nya support 3G
             - touch_screen: Layarnya Bisa di sentuh Atau tidak
             - wifi: Memiliki Jaringan WIFI atau Tidak
             - Price range: label dari kisaran harga
             
             Index
             Output Dari Dataset ini merupakan sebuah index yaitu : 0,1,2,3, 
             dimana dari 4 index ini di kategorikan sebagai berikut
             > 0 - Low Cost
             > 1 - Medium Cost
             > 2 - High Cost
             > 3 - Very High Cost
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
