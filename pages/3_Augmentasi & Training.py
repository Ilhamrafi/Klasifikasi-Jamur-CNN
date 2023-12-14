import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Menonaktifkan peringatan st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Augmentasi & Training Model (CNN)")

train_data_dir = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP/train'
test_data_dir = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP/test'

genus_folders = os.listdir(train_data_dir)

image_counts_train = {}
image_counts_test = {}

for genus_folder in genus_folders:
    genus_path_train = os.path.join(train_data_dir, genus_folder)
    genus_path_test = os.path.join(test_data_dir, genus_folder)
    
    if os.path.isdir(genus_path_train):
        image_counts_train[genus_folder] = len(os.listdir(genus_path_train))
    if os.path.isdir(genus_path_test):
        image_counts_test[genus_folder] = len(os.listdir(genus_path_test))

df_image_counts_train = pd.DataFrame.from_dict(image_counts_train, orient='index', columns=['Jumlah Gambar Training'])
df_image_counts_test = pd.DataFrame.from_dict(image_counts_test, orient='index', columns=['Jumlah Gambar Testing'])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Training")
    st.write(df_image_counts_train)

with col2:
    st.subheader("Data Testing")
    st.write(df_image_counts_test)

st.subheader("Tuning Parameter:")
target_size = st.slider("Target Size", min_value=100, max_value=500, step=50, value=150)
batch_size = st.slider("Batch Size", min_value=32, max_value=256, step=32, value=64)
epochs = st.slider("Jumlah epoch", min_value=1, max_value=100, value=1)

st.write("Jumlah epoch:", epochs)

st.info("Klik start untuk memulai proses training!")

if st.button("Start Augmentasi & Training"):
    with st.spinner("Proses training sedang berlangsung..."):


        # Contoh penggunaan
        train_dir = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP/train'
        test_dir = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP/test'
        image_path = 'C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/dataset_PP/train/Pleurotus/20230316_102744.jpg'

        image_data = ImageData(train_dir, test_dir)

        # Menampilkan gambar sebelum preprocessing dan matriks originalnya
        original_img_tensor = image_data.plot_original_image(image_path)

        pic = image_data.generate_images(original_img_tensor)
        image_data.plot_images(pic)

        train_generator = image_data.generate_train_generator()
        test_generator = image_data.generate_val_generator()

        # Mencetak matriks gambar dari img_tensor yang sudah rescaled
        rescaled_img_tensor = original_img_tensor / 255.0
        # print('Rescaled Image Shape:', rescaled_img_tensor.shape)
        st.write('Matrix setelah rescale:')
        st.write(rescaled_img_tensor)

        # Menampilkan matriks gambar dari train_generator
        for data_batch, labels_batch in train_generator:
            st.write('Data shape:', data_batch.shape)  # bentuk matriks gambar
            st.write('Labels shape:', labels_batch.shape)  # bentuk matriks label
            break

        # Menampilkan matriks gambar dari test_generator
        for data_batch, labels_batch in test_generator:
            st.write('Data shape:', data_batch.shape)  # bentuk matriks gambar
            st.write('Labels shape:', labels_batch.shape)  # bentuk matriks label
            break
        #definisikan model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(target_size, target_size, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.summary()

        model.compile(loss="categorical_crossentropy",
                        optimizer='Adam',
                        metrics=["accuracy"])

        # Menampilkan data
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Train")
            st.write("Jumlah gambar: " + str(train_generator.samples))
            st.write("Jumlah kelas: " + str(train_generator.num_classes))
            st.write("Batch size: " + str(train_generator.batch_size))

        with col2:
            st.subheader("Data Testing")
            st.write("Jumlah gambar testing: " + str(test_generator.samples))
            st.write("Jumlah kelas testing: " + str(test_generator.num_classes))
            st.write("Batch size: " + str(test_generator.batch_size))
        
        # Menyiapkan data untuk grafik batang
        train_data = pd.DataFrame({"Data": ["Train"], "Jumlah Gambar": [train_generator.samples]})
        test_data = pd.DataFrame({"Data": ["Test"], "Jumlah Gambar": [test_generator.samples]})
        combined_data = pd.concat([train_data, test_data])

        # Membuat grafik batang
        fig, ax = plt.subplots()
        ax.bar(combined_data['Data'], combined_data['Jumlah Gambar'], width=0.6)

        # Menambahkan label pada sumbu y
        ax.set_ylabel('Jumlah Gambar')

        # Menampilkan judul grafik
        plt.title('Perbandingan Jumlah Gambar Training dan Testing')

        st.pyplot(fig)

        # Menampilkan proses augmentasi
        st.header("Augmented Images")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Train Images")
            for i in range(5):
                image, label = train_generator.next()
                st.image(image[0], caption=f"Label: {label[0]}")

        with col4:
            st.subheader("Testing Images")
            for i in range(5):
                image, label = test_generator.next()
                st.image(image[0], caption=f"Label: {label[0]}")
        
        st.header("Training Progress")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        epoch_time_text = st.empty()

        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []

        start_time = time.time()

        for epoch in range(epochs):
            st.subheader(f"Epoch {epoch + 1}/{epochs}")
            progress_text.text("0%")
            epoch_time_text.empty()

            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_data=test_generator,
                validation_steps=test_generator.samples // test_generator.batch_size,
                epochs=1,
                verbose=0
            )

            train_accuracy.append(history.history['accuracy'][0])
            test_accuracy.append(history.history['val_accuracy'][0])
            train_loss.append(history.history['loss'][0])
            test_loss.append(history.history['val_loss'][0])

            progress_value = int(((epoch + 1) / epochs) * 100)
            progress_bar.progress(progress_value)
            progress_text.text(f"{progress_value}%")

            epoch_time = time.time() - start_time
            epoch_time_text.text(f"Elapsed Time: {epoch_time:.2f} seconds")

            # Tampilkan informasi training setiap epoch
            st.write("Train Accuracy:", train_accuracy[-1])
            st.write("Test Accuracy:", test_accuracy[-1])
            st.write("Train Loss:", train_loss[-1])
            st.write("Test Loss:", test_loss[-1])

            # Tampilkan persentase yang berjalan untuk setiap epoch
            st.write(f"Persentase Selesai: {progress_value}%")

            # Tampilkan progress bar di setiap epoch
            st.progress(progress_value / 100)

        st.success("Training Selesai!")


        st.header("Training Metrics")

        # Plot accuracy
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy')
        ax.plot(range(1, epochs + 1), test_accuracy, label='Testing Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Testing Accuracy')
        ax.legend()
        st.pyplot(fig)

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        ax.plot(range(1, epochs + 1), test_loss, label='Testing Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Testing Loss')
        ax.legend()
        st.pyplot(fig)

        # # Menampilkan ringkasan model
        # st.header("Model Summary")
        # model.summary(print_fn=st.write)

        # Simpan model
        # model.save("model.h5")
        # st.success("Model berhasil disimpan dengan nama 'model.h5'")