# imports
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

main_directory = Path('C:/main_directory') 

#print each class from the dataset and the total number of images each class contains
"""

for label in main_directory.iterdir():
    if label.is_dir():
        image_count = len(list(label.glob('*.jpg')))
        print(f"Label: '{label.name}': contains {image_count} images")


# print the first image of each class with their label name
for subdirectory in main_directory.iterdir():
    if subdirectory.is_dir():
        image_paths = list(subdirectory.glob('*.jpg'))
        if image_paths:
            first_image_path = image_paths[0]
            image = Image.open(first_image_path)
            plt.imshow(image)
            plt.axis('off')
            plt.title(subdirectory.name)  # Display the label name as the image title
            plt.show()
"""
#create a dataset
# Load data using a Keras utility
#using a 80:20 split for the train_dataset and the validation_dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    main_directory,
    labels='inferred',      #labels inferred from the subdirectory names
    image_size=(64, 64),  #image height and width
    batch_size=32,
    seed=123,
    shuffle=True,
    validation_split=0.2,
    subset='training'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    main_directory,
    labels='inferred',
    image_size=(64, 64),
    batch_size=32,
    seed=456,
    shuffle=True,
    validation_split=0.2,
    subset='validation'
)

class_names = train_dataset.class_names
print(class_names)

"""

#print the first 9 images from the train_dataset
class_names = train_dataset.class_names
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs = axs.flatten()
for images, labels in train_dataset.take(1):
    for i in range(9):
        axs[i].imshow(images[i].numpy().astype("uint8"))
        axs[i].set_title(class_names[labels[i]])
        axs[i].axis('off')
plt.tight_layout()
plt.show()
"""

# Configure the dataset for performance using buffered prefetching
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# data augmentation

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(64,
                                       64,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

"""
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
"""

num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 15
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)
#visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Predict on new data
image_path = 'C:/sandstorm.jpg' 
img_height, img_width = 64, 64

img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
predicted_label = class_names[np.argmax(score)]
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted label: {predicted_label}')
plt.show()