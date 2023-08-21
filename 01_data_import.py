import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = 'data'
train_valid_dir = os.path.join(data_dir, 'train-valid')
test_dir = os.path.join(data_dir, 'test')


def load_and_preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)  # Resize to the desired target size
    image = np.array(image) / 255.0    # Normalize pixel values to [0, 1]
    return image


def load_image_dataset(dataset_dir):
    data = []
    labels = []
    classes = os.listdir(dataset_dir)
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = load_and_preprocess_image(image_path, target_size)
                    data.append(image)
                    labels.append(1 if class_name == 'positive' else 0)  # Convert labels to binary
                    
    return data, labels

target_size = (1024, 1024)  # Desired target size for images


train_data, train_labels = load_image_dataset(train_valid_dir)
test_data, test_labels = load_image_dataset(test_dir)

train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.2, shuffle=True, random_state=1)


# split all dataset 10% test, 90% train (after that the 90% train will split to 20% validation and 80% train

# now we have 10% test, 72% training and 18% validation

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(valid_data))
print("Number of test samples:", len(test_data))
print("Number of total samples:", len(train_data)+len(test_data)+len(valid_data))

# Create a pretrained ResNet-50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification, so sigmoid activation

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Prepare data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Train the model
batch_size = 32
epochs = 10
steps_per_epoch = len(train_data) // batch_size

model.fit(data_augmentation.flow(np.array(train_data), np.array(train_labels), batch_size=batch_size),
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(np.array(valid_data), np.array(valid_labels)))


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(np.array(test_data), np.array(test_labels))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

