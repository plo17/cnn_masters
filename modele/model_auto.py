from tensorflow import keras
from tensorflow.keras import Sequential
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

from input_data import class_names, train_ds, val_ds
from augmentation import data_augmentation

def build_model(hp):
    model = keras.Sequential([
        data_augmentation,
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=16, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu',
            input_shape=(128, 128, 3)
        ),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
            activation='relu'
        ),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld'
)

tuner.search_space_summary()

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[keras.callbacks.TensorBoard(log_dir="callback")]
)

best_model = tuner.get_best_models(num_models=1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hyperparameters.values)

#best_model.save('best_model_v1.h5')

