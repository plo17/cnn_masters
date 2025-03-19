import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from input_data import class_names


model = tf.keras.models.load_model('modele/model_2.keras')
new_images_dir = "C:/Users/magda/Desktop/validation"


def load_and_preprocess_image(img_path, target_size=(128, 128)):
    """
    Ładuje i przetwarza zdjęcie do formatu, który może być użyty przez model.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



for img_name in os.listdir(new_images_dir):
    img_path = os.path.join(new_images_dir, img_name)

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print(f"Pominięto plik (nie jest obrazem): {img_name}")
        continue

    try:
        img_array = load_and_preprocess_image(img_path)

        predictions = model.predict(img_array)[0]
        print("Min:", np.min(img_array), "Max:", np.max(img_array))

        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Sortowanie i pobranie top 3
        top_3_probs = predictions[top_3_indices]  # Pobranie prawdopodobieństw

        print(f"Zdjęcie: {img_name}")
        for i, idx in enumerate(top_3_indices):
            print(f"{i + 1}. {class_names[idx]}: {round(top_3_probs[i] * 100, 2)}%")
        print("-" * 50)



    except Exception as e:
        print(f"Błąd podczas przetwarzania zdjęcia {img_name}: {e}")
