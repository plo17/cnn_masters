from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from input_data import class_names, train_ds, val_ds
from augmentation import data_augmentation
import tensorflow as tf
import keras_tuner as kt
from confusion_matrix import log_confusion_matrix
from sql_save import training_results


num = 4
run_name = f"model_{num}"
log_dir = f"./logs/{run_name}"

file_writer_cm = tf.summary.create_file_writer(log_dir)
def build_model(hp):
    dropout_rate = hp.Choice('dropout', [0.1, 0.3, 0.5])

    model = keras.Sequential([
        keras.layers.Input(shape=(128, 128, 3)),
        data_augmentation,

        keras.layers.Conv2D(hp.Choice('filters', [32, 64, 128]), (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Conv2D(hp.Choice('filters', [32, 64, 128]), (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Conv2D(hp.Choice('filters', [32, 64, 128]), (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Flatten(),
        keras.layers.Dense(hp.Choice('dense_units', [64, 128, 256]), activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # Kompilacja modelu z dynamicznym learning_rate
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.01, 0.1])),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

#KerasTuner do doboru hiperparametrów
tuner = kt.GridSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='hyperparameter_tuning',
    project_name=f'CNN_tuning_{num}'
)

# uruchomienie
tuner.search(train_ds, validation_data=val_ds, epochs=20, verbose=1)

# pobranie najlepszych hiperparametrów
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Najlepsze hiperparametry: dropout={best_hps.get('dropout')}filters={best_hps.get('filters')}, dense_units={best_hps.get('dense_units')}, learning_rate={best_hps.get('learning_rate')}")


model = build_model(best_hps)

# Trening modelu z najlepszymi ustawieniami
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
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

best_epoch = early_stop_callback.stopped_epoch if early_stop_callback.stopped_epoch else len(history.history['loss'])

# Pobranie wyników z najlepszej epoki
train_acc = history.history['accuracy'][best_epoch - 1]
val_acc = history.history['val_accuracy'][best_epoch - 1]
train_loss = history.history['loss'][best_epoch - 1]
val_loss = history.history['val_loss'][best_epoch - 1]


training_results(f"model_{num}", best_epoch, train_acc, val_acc, train_loss, val_loss)
print("Wyniki zapisane w bazie!")

print(f"Model zatrzymał trening na epoce: {len(history.history['loss'])}")
model.save(f'model_{num}.keras')


def save_model_summary(model, filename=f"model_summary_{num}.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        # Przekierowanie outputu do pliku
        model.summary(print_fn=lambda x: f.write(x + "\n"))

save_model_summary(model)



