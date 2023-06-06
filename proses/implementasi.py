import streamlit as st
import joblib

Grade = [0, 1, 2, 3]


def nb(data_input_baru):
    model = joblib.load('model/modelNB.pkl')

    st.write('Hasil Prediksi')
    Y_pred_scaler = model.predict(data_input_baru)
    prediction_index = int(Y_pred_scaler[0][-1])

    st.success(
        f'Berdasarkan data yang sudah diinputkan, termasuk dalam kelas: {Grade[prediction_index-1]}')
