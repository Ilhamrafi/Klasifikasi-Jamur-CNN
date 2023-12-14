import streamlit as st
import cv2 
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model("C:/Users/ASUS/OneDrive/Dokumen/Informatika Semester 6/Proyek Profesional/Image_Classification/my_model.hdf5")
    return model

def import_and_predict(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    result = model.predict(image)
    return result

def run_object_detection():
    st.title("Detection Object") 
    st.write("Upload gambar untuk diklasifikasikan di bawah:")
    file = st.file_uploader("Choose the file", type=['jpg', 'jpeg', 'png'])
    use_camera = st.checkbox("Gunakan Kamera")

    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        model = load_model()
        predictions = import_and_predict(image, model)
        class_names = ['Jamur genus pleurotus', 'Jamur genus amanita', 'Bukan jamur']
        result = "Gambar yang anda upload merupakan gambar: " + class_names[np.argmax(predictions)]

        # Menampilkan probabilitas
        if np.argmax(predictions) == 0:
            result += ", genus ini dapat dikonsumsi"
            st.success(result) # Tampilkan teks dengan warna hijau jika dapat dikonsumsi
        elif np.argmax(predictions) == 2:
            result += ", Silahkan upload gambar jamur!"
            st.warning(result)  # Tampilkan teks dengan warna kuning jika bukan jamur
        else:
            result += ", genus ini tidak dapat dikonsumsi"
            st.warning(result) # Tampilkan teks dengan warna kuning jika tidak dapat dikonsumsi

        # Menampilkan probabilitas dalam bentuk persen
        st.write("Probabilitas Klasifikasi:")
        for i in range(len(class_names)):
            probability = predictions[0][i] * 100
            st.info(f"{class_names[i]}: {probability:.2f}%")

    if use_camera:
        run_camera()

def run_camera():
    model = load_model()
    video_capture = cv2.VideoCapture(0)
    button_pressed = False

    while True:
        ret, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Menampilkan frame dari kamera
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Menggunakan tombol untuk mengambil gambar dari kamera
        if not button_pressed and st.button("Ambil Gambar"):
            button_pressed = True
            image = Image.fromarray(frame_rgb)
            predictions = import_and_predict(image, model)
            class_names = ['Jamur genus pleurotus', 'Jamur genus amanita', 'Bukan jamur']
            result = "Gambar yang diambil merupakan gambar: " + class_names[np.argmax(predictions)]

            # Menampilkan probabilitas
            if np.argmax(predictions) == 0:
                result += ", genus ini dapat dikonsumsi"
                st.success(result)  # Tampilkan teks dengan warna hijau jika dapat dikonsumsi
            elif np.argmax(predictions) == 2:
                result += ", Silahkan upload gambar jamur!"
                st.warning(result)  # Tampilkan teks dengan warna kuning jika bukan jamur
            else:
                result += ", genus ini tidak dapat dikonsumsi"
                st.warning(result)  # Tampilkan teks dengan warna kuning jika tidak dapat dikonsumsi

            # Menampilkan probabilitas dalam bentuk persen
            st.write("Probabilitas Klasifikasi:")
            for i in range(len(class_names)):
                probability = predictions[0][i] * 100
                st.info(f"{class_names[i]}: {probability:.2f}%")
            break

    video_capture.release()

def main():
    run_object_detection()

if __name__ == "__main__":
    main()