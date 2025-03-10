import tensorflow as tf
import pathlib
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# path to images
directory = "C:/Users/magda/Desktop/eggs"
data_dir = pathlib.Path(directory)

batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names
print("CLASS NAMES\n", class_names)
# Check the shape of an image batch
image_batch, labels_batch = next(iter(train_ds))
print(image_batch.shape)


#Standardize the data
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



for images, labels in val_ds.take(1):
    print("Batch shape (images):", images.shape)
    print("Batch shape (labels):", labels.shape)
    print("Example labels:", labels.numpy())



"""# Visualize one image per class
fig, axes = plt.subplots(2, len(class_names) // 2 + len(class_names) % 2, figsize=(15, 6))
axes = axes.flatten()
selected_images = {}

for images, labels in train_ds:
    for img, label in zip(images, labels):
        class_name = class_names[label.numpy()]
        if class_name not in selected_images:
            selected_images[class_name] = img
        if len(selected_images) == len(class_names):
            break
    if len(selected_images) == len(class_names):
        break

for ax, (class_name, img) in zip(axes, selected_images.items()):
    ax.imshow(img.numpy())
    ax.set_title(class_name)
    ax.axis("off")
# Hide unused subplots
for ax in axes[len(selected_images):]:
    ax.axis("off")

plt.tight_layout()
plt.show()"""