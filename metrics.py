import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow import keras
from input_data import val_ds

model = keras.models.load_model('modele/model_1.keras')

def extract_labels(dataset):
    labels = []
    for _, label_batch in dataset:
        labels.extend(label_batch.numpy())
    return np.array(labels)

y_true = extract_labels(val_ds)


# Make predictions
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)


# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')


# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


