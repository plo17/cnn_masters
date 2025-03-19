from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from input_data import class_names, train_ds, val_ds
from augmentation import data_augmentation
import tensorflow as tf
from confusion_matrix import log_confusion_matrix
from sql_save import training_results

num = "1"
run_name = f"model_{num}"
log_dir = f"./logs/{run_name}"
file_writer_cm = tf.summary.create_file_writer(log_dir)

def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(128, 128, 3)),
        data_augmentation,

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.1),

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
    log_dir=log_dir,
    histogram_freq=1
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2,
    restore_best_weights=True
)



epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
    callbacks=[tensorboard_callback, early_stop_callback]
)




best_epoch = early_stop_callback.stopped_epoch if early_stop_callback.stopped_epoch else len(history.history['loss'])
log_confusion_matrix(best_epoch, model, val_ds, class_names, file_writer_cm)

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
        model.summary(print_fn=lambda x: f.write(x + "\n"))


save_model_summary(model)