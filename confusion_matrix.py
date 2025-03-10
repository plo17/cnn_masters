import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm = cm.astype("float") / row_sums

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
    The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def log_confusion_matrix(epoch, model, val_ds, class_names, file_writer_cm):
    # Use the model to predict the values from the validation dataset.
    val_images = []
    val_labels = []

    # Extract images and labels from val_ds
    for images, labels in val_ds:
        val_images.append(images)
        val_labels.append(labels)

    val_images = tf.concat(val_images, axis=0)
    val_labels = tf.concat(val_labels, axis=0)

    val_pred_raw = model.predict(val_images)
    val_pred = np.argmax(val_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = confusion_matrix(val_labels, val_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

