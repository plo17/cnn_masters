import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from input_data import class_names  # Załaduj listę nazw klas


model = tf.keras.models.load_model('modele/model_1.keras')
new_images_dir = "C:/Users/magda/Desktop/validation"


def load_and_preprocess_image(img_path, target_size=(128, 128)):
    """
    Ładuje i przetwarza zdjęcie do formatu, który może być użyty przez model.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array



for img_name in os.listdir(new_images_dir):
    img_path = os.path.join(new_images_dir, img_name)

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print(f"Pominięto plik (nie jest obrazem): {img_name}")
        continue

    try:
        img_array = load_and_preprocess_image(img_path)

        predictions = model.predict(img_array)[0]
        top_3_indices = np.argsort(predictions)[-3:][::-1]


        print(f"Zdjęcie: {img_name}")
        for i, idx in enumerate(top_3_indices):
            class_name = class_names[idx]
            probability = round(predictions[idx] * 100, 2)
            print(f"{i + 1}. {class_name}: {probability}%")
        print("-" * 50)

    except Exception as e:
        print(f"Błąd podczas przetwarzania zdjęcia {img_name}: {e}")
