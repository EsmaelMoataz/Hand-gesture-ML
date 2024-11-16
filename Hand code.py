import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Add
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
import time
from skimage import filters

# Set matplotlib backend to 'Agg'
matplotlib_use('Agg')

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Local paths
local_train_dir = '/Users/esmaelmoataz/Documents/Machine learning/sample/train'
local_val_dir = '/Users/esmaelmoataz/Documents/Machine learning/sample/val'

# Gesture names
gesture_names = ['dislike', 'like', 'stop', 'peace', 'rock', 'ok']
num_classes = len(gesture_names)

# Function to preprocess individual photos with edge detection
def preprocess_photo(photo_path):
    img_string = tf.io.read_file(photo_path)
    img = tf.image.decode_image(img_string, channels=1)
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, (128, 128))
    img.set_shape([128, 128, 1])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.numpy_function(filters.sobel, [img], tf.float32)  # Apply edge detection
    img = tf.reshape(img, [128, 128, 1])  # Ensure the shape after applying edge detection
    return img

# Function to load folder paths
def load_folder_paths(base_dir, gesture_names):
    folder_paths = []
    for gesture_name in gesture_names:
        gesture_dir = os.path.join(base_dir, gesture_name)
        if os.path.isdir(gesture_dir):
            subfolders = [os.path.join(gesture_dir, d) for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]
            for subfolder in subfolders:
                folder_paths.append((subfolder, gesture_name))
    return folder_paths

# Function to count files in each folder
def count_files_in_folders(folder_paths):
    file_counts = {}
    for folder, gesture_name in folder_paths:
        file_counts[gesture_name] = file_counts.get(gesture_name, 0) + len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    return file_counts

# Verify the counts of files for each gesture
train_folder_paths = load_folder_paths(local_train_dir, gesture_names)
val_folder_paths = load_folder_paths(local_val_dir, gesture_names)

print("Verifying file counts before splitting:")
train_file_counts = count_files_in_folders(train_folder_paths)
val_file_counts = count_files_in_folders(val_folder_paths)
for gesture_name in gesture_names:
    print(f"Gesture: {gesture_name}, Train files: {train_file_counts.get(gesture_name, 0)}, Validation files: {val_file_counts.get(gesture_name, 0)}")

# Function to load photo paths and labels from selected folders
def load_photo_paths_from_folders(folder_paths, gesture_label):
    photo_paths = []
    labels = []
    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    photo_paths.append(os.path.join(root, file))
                    labels.append(gesture_label)
    return photo_paths, labels

# Function to create a dataset
def create_dataset(photo_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((photo_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_photo(x), tf.one_hot(y, depth=num_classes)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=len(photo_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat the dataset indefinitely
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Parameters
input_shape = (128, 128, 1)
batch_size = 32

# Regularization parameters
l2_weight_conv = 0.00001
l2_weight_dense = 0.00001

# Create main validation dataset
main_val_photo_paths, main_val_labels = [], []
for gesture_name in gesture_names:
    gesture_dir = os.path.join(local_val_dir, gesture_name)
    if os.path.isdir(gesture_dir):
        paths, labels = load_photo_paths_from_folders([gesture_dir], gesture_names.index(gesture_name))
        main_val_photo_paths.extend(paths)
        main_val_labels.extend(labels)

main_val_dataset = create_dataset(main_val_photo_paths, main_val_labels, batch_size)
main_validation_steps = len(main_val_photo_paths) // batch_size

# Define custom callbacks
class CustomModelCheckpoint(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        h5_path = os.path.join(self.save_dir, f'model_2d_epoch_{epoch + 1:02d}_val_accuracy_{logs["val_accuracy"]:.2f}.h5')
        keras_path = os.path.join(self.save_dir, f'model_2d_epoch_{epoch + 1:02d}_val_accuracy_{logs["val_accuracy"]:.2f}.keras')
        self.model.save(h5_path)
        self.model.save(keras_path)
        print(f"Saved model at {h5_path} and {keras_path}")

checkpoint_callback = CustomModelCheckpoint(local_train_dir)

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"\nEpoch {epoch + 1}: Learning Rate is {lr:.6f}")

class MainValidationLogger(Callback):
    def __init__(self, main_val_data, main_validation_steps):
        super().__init__()
        self.main_val_data = main_val_data
        self.main_validation_steps = main_validation_steps

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nStarting main validation for epoch {epoch + 1}")
        start_time = time.time()
        main_val_loss, main_val_accuracy = self.model.evaluate(self.main_val_data, steps=self.main_validation_steps, verbose=1)
        end_time = time.time()
        print(f"\nEpoch {epoch + 1}: Main Validation Loss: {main_val_loss:.4f}, Main Validation Accuracy: {main_val_accuracy:.4f}")
        print(f"Main validation took {end_time - start_time:.2f} seconds.")
        logs['main_val_loss'] = main_val_loss
        logs['main_val_accuracy'] = main_val_accuracy

# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True, write_images=True)

# Define ResNet30 architecture with two dense layers
def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = -1
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, kernel_regularizer=l2(l2_weight_conv), name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(l2_weight_conv), name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_weight_conv), name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)
    x = Conv2D(filters, 1, kernel_regularizer=l2(l2_weight_conv), name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '_3_bn')(x)
    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x

def ResNet30(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 7, strides=2, padding='same', kernel_regularizer=l2(l2_weight_conv), name='conv1_conv')(inputs)
    x = BatchNormalization(axis=-1, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)
    x = resnet_block(x, 32, conv_shortcut=True, name='conv2_block1')
    x = resnet_block(x, 32, conv_shortcut=False, name='conv2_block2')
    x = resnet_block(x, 32, conv_shortcut=False, name='conv2_block3')
    x = resnet_block(x, 64, stride=2, conv_shortcut=True, name='conv3_block1')
    x = resnet_block(x, 64, conv_shortcut=False, name='conv3_block2')
    x = resnet_block(x, 64, conv_shortcut=False, name='conv3_block3')
    x = resnet_block(x, 64, conv_shortcut=False, name='conv3_block4')
    x = resnet_block(x, 128, stride=2, conv_shortcut=True, name='conv4_block1')
    x = resnet_block(x, 128, conv_shortcut=False, name='conv4_block2')
    x = resnet_block(x, 128, conv_shortcut=False, name='conv4_block3')
    x = resnet_block(x, 128, conv_shortcut=False, name='conv4_block4')
    x = resnet_block(x, 300, stride=2, conv_shortcut=True, name='conv5_block1')
    x = resnet_block(x, 300, conv_shortcut=False, name='conv5_block2')
    x = resnet_block(x, 300, conv_shortcut=False, name='conv5_block3')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation='relu', kernel_regularizer=l2(l2_weight_dense), name='fc1')(x)
    x = Dropout(0.1)(x)
    x = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_weight_dense), name='fc2')(x)
    model = Model(inputs, x, name='resnet30_2d')
    return model

# Initialize the model
model_2d = ResNet30(input_shape, num_classes)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model_2d.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training and validation datasets
train_photo_paths, train_labels = [], []
for gesture_name in gesture_names:
    gesture_dir = os.path.join(local_train_dir, gesture_name)
    if os.path.isdir(gesture_dir):
        paths, labels = load_photo_paths_from_folders([gesture_dir], gesture_names.index(gesture_name))
        train_photo_paths.extend(paths)
        train_labels.extend(labels)

train_dataset = create_dataset(train_photo_paths, train_labels, batch_size)
steps_per_epoch = len(train_photo_paths) // batch_size

# Training the model
history_2d = model_2d.fit(
    train_dataset,
    validation_data=main_val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=main_validation_steps,
    epochs=3500,
    callbacks=[
        checkpoint_callback,
        LearningRateLogger(),
        tensorboard_callback,
        MainValidationLogger(main_val_data=main_val_dataset, main_validation_steps=main_validation_steps)
    ]
)

# Final evaluation on the independent validation set
final_val_loss, final_val_accuracy = model_2d.evaluate(main_val_dataset, steps=main_validation_steps)
print(f'Final validation set performance - Loss: {final_val_loss}, Accuracy: {final_val_accuracy}')
