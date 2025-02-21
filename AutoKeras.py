import numpy as np
import tensorflow as tf
import autokeras as ak

from input_data import train_ds, val_ds
from augmentation import data_augmentation

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

clf = ak.ImageClassifier(overwrite=True, max_trials=2)
clf.fit(train_ds, validation_data=val_ds, epochs=1)
predicted = clf.predict(val_ds)

print(predicted)

loss, accuracy = clf.evaluate(val_ds)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")