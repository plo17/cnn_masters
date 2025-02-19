import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from input_data import val_ds
from modele import model_v1
import keras

model = keras.models.load_model('model.h5')

# After training the model, make predictions
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = val_ds.classes  # True labels from validation dataset

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')  # You can change average as per your needs
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
specificity = tn / (tn + fp)  # Specificity calculation

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")
