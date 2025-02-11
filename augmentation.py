from tensorflow.keras import layers
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  # Obrót w poziomie i pionie
    layers.RandomRotation(0.4),                    # Losowa rotacja
    layers.RandomZoom(0.2),                        # Losowe przybliżanie/oddalanie
    layers.RandomContrast(0.2),                    # Zmiana kontrastu
])