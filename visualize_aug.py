import matplotlib.pyplot as plt
from math import ceil
from input_data import train_ds, class_names

# Number of classes and calculation of grid size
num_classes = len(class_names)
cols = 3  # Number of columns
rows = ceil(num_classes / cols)  # Number of rows needed to fit all classes

# Prepare the plot
fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(cols * 3, rows * 3))

# Prepare storage for images and labels
images_to_plot = {}

# Iterate through batches to find all classes
for images, labels in train_ds:
    for img, label in zip(images, labels):
        class_idx = int(label)
        if class_idx not in images_to_plot:
            images_to_plot[class_idx] = img
        if len(images_to_plot) == num_classes:  # Stop when we have images for all classes
            break
    if len(images_to_plot) == num_classes:
        break

# Display the images
for idx, (class_idx, img) in enumerate(images_to_plot.items()):
    i, j = divmod(idx, cols)  # Calculate the position in the grid
    ax[i][j].imshow(img.numpy().astype("uint8"))
    ax[i][j].set_title(class_names[class_idx])
    ax[i][j].axis('off')

# Hide empty axes if they exist
for idx in range(num_classes, rows * cols):
    i, j = divmod(idx, cols)
    ax[i][j].axis('off')

plt.tight_layout()
plt.show()
