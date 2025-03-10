from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from input_data import class_names, train_ds, val_ds
from augmentation import data_augmentation
import tensorflow as tf
from confusion_matrix import log_confusion_matrix


num = 1
run_name = f"model_{num}"
log_dir = f"./logs/{run_name}"
file_writer_cm = tf.summary.create_file_writer(log_dir)

def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(128,128,3)),
        #data_augmentation,

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

model=get_model()

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)
model.summary()

tensorboard_callback = TensorBoard(
    log_dir=log_dir, histogram_freq=1
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)

cm_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: log_confusion_matrix(epoch, model, val_ds, class_names, file_writer_cm)
)

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
    callbacks=[tensorboard_callback, early_stop_callback, cm_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Model zatrzymał trening na epoce: {len(history.history['loss'])}")
model.save(f'model_{num}.keras')


def save_model_summary(model, filename=f"model_summary_{num}.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        # Przekierowanie outputu do pliku
        model.summary(print_fn=lambda x: f.write(x + "\n"))

# Przykładowe wywołanie
save_model_summary(model)