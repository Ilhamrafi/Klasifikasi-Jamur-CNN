import streamlit as st
from PIL import Image, ImageOps


st.title("CNN CLASSIFICATION")
st.markdown("<h2 style='text-align: center; font-size:20px;'>IMPLEMENTASI DEEP LEARNING UNTUK KLASIFIKASI CITRA JAMUR MENGGUNAKAN ALGORITMA CONVOLUTIONAL NEURAL NETWORK</h2>", unsafe_allow_html=True)

# Membaca gambar dari file
gambar1 = Image.open('C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/CODING/data_jamur/train/Agaricus/001_2jP9N_ipAo8.jpg').resize((300,300))
gambar2 = Image.open('C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/CODING/data_jamur/train/Amanita/144_FnkvkEdRMr0.jpg').resize((300,300))
gambar3 = Image.open('C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/CODING/data_jamur/train/Boletus/0018_PcgJi7QNZ_w.jpg').resize((300,300))
gambar4 = Image.open('C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/CODING/data_jamur/train/Cortinarius/872_5mcRTuMgqkc.jpg').resize((300,300))

# Menampilkan gambar dalam format 2 baris 2 kolom
col1, col2 = st.columns(2)
col1.image(gambar1, caption='Jamur Genus Agaricus (edible)', use_column_width=True)
col2.image(gambar2, caption='Jamur Genus Amanita (non-edible)', use_column_width=True)

col3, col4 = st.columns(2)
col3.image(gambar3, caption='Jamur Boletus (edible)', use_column_width=True)
col4.image(gambar4, caption='Jamur Genus Cortinarius (non-edible)', use_column_width=True)