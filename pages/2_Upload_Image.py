import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import os

# inisialisasi variabel global
uploaded_image = None
uploaded_image_path = None

st.title("Upload or Capture Image")
uploaded_file = None
camera_image = None
image = None
label = ""

# Menentukan path folder
base_dir = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

label_dirs = ['Pleurotus', 'Amanita', 'random_images']
save_dirs = [train_dir, test_dir, valid_dir]

# Fungsi untuk membuat direktori jika belum ada
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Bagian upload gambar dari direktori lokal
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    print("Gambar berhasil diunggah.")
    uploaded_image = Image.open(uploaded_file)
    st.write("Selected Image:")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    label = st.text_input("Input label for the image:")

    label_dir = st.selectbox("Select label directory", label_dirs)
    save_dir = st.selectbox("Select save directory", save_dirs, index=0)

    if st.button("Save Uploaded Image"):
        if uploaded_file is not None and label:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            file_name = label + file_extension

            if label_dir == 'Pleurotus':
                save_subdir = os.path.join(save_dir, 'Pleurotus')
            elif label_dir == 'Amanita':
                save_subdir = os.path.join(save_dir, 'Amanita')
            elif label_dir == 'random_images':
                save_subdir = os.path.join(save_dir, 'random_images')

            create_directory_if_not_exists(save_subdir)

            img = Image.open(uploaded_file)
            uploaded_image_path = os.path.join(save_subdir, file_name)
            img.save(uploaded_image_path)
            uploaded_image = img  # Simpan gambar ke dalam variabel uploaded_image
            st.image(uploaded_image, caption=label, use_column_width=True)
            st.success("Image saved to directory upload!")
        else:
            st.warning("Please choose an image file and provide a label.")