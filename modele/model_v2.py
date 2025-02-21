from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard

from input_data import class_names, train_ds, val_ds
from augmentation import data_augmentation

import keras

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)


def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
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

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="callback", histogram_freq=1,
)

epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
    callbacks=[tensorboard_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#model.save('model_v2.h5')

