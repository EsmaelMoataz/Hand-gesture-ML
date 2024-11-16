import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Add
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import time

# Set matplotlib backend to 'Agg'
matplotlib_use('Agg')

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Local paths
local_train_dir = '/Users/esmaelmoataz/Documents/Machine learning/Hand gesture/train'
local_val_dir = '/Users/esmaelmoataz/Documents/Machine learning/Hand gesture/val'

# Gesture names
gesture_names = ['dislike', 'like', 'stop']  # Add other gesture names as needed
num_classes = len(gesture_names)

# Function to preprocess individual photos
def preprocess_photo(photo_path):
    img_string = tf.io.read_file(photo_path)
    img = tf.image.decode_image(img_string, channels=1)
    img.set_shape([None, None, 1])  # Set shape after decoding to include the channel dimension
    img = tf.image.resize(img, (128, 128))  # Resize image
    img.set_shape([128, 128, 1])  # Explicitly set the final shape after resizing
    img = tf.cast(img, tf.float32) / 255.0
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
input_shape = (128, 128, 1)  # Updated to match new image size
batch_size = 32  # Adjust batch size as needed

# Regularization parameters
l2_weight_conv = 0
l2_weight_dense = 0

# Extract folder paths and labels
all_train_folder_paths = [fp[0] for fp in train_folder_paths]
all_train_folder_labels = [gesture_names.index(fp[1]) for fp in train_folder_paths]

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

# Custom function to handle stratified K-Fold
def stratified_kfold(all_folder_paths, all_folder_labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(all_folder_paths, all_folder_labels):
        train_paths = [all_folder_paths[i] for i in train_index]
        train_labels = [all_folder_labels[i] for i in train_index]
        val_paths = [all_folder_paths[i] for i in val_index]
        val_labels = [all_folder_labels[i] for i in val_index]
        yield train_paths, train_labels, val_paths, val_labels

# Initialize lists to store metrics for each fold
fold_val_accuracies = []
fold_val_losses = []

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

# Custom callback for real-time graph
class RealTimeGraph(Callback):
    def __init__(self, expected_accuracy):
        super().__init__()
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.colors = plt.cm.viridis(np.linspace(0, 1, 20))  # Use a colormap with 20 different colors
        self.epoch_data = []
        self.expected_accuracy = expected_accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Store epoch data
        self.epoch_data.append((epoch, logs['val_accuracy']))
        self.ax.clear()
        self.ax.set_xlim(0, len(self.epoch_data) + 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Validation Accuracy')

        # Plot data points
        epochs = [e for e, _ in self.epoch_data]
        accuracies = [acc for _, acc in self.epoch_data]
        self.ax.plot(epochs, accuracies, color='blue', linestyle='-', linewidth=2, label='Actual Accuracy')

        # Plot hypothesis line
        expected_accuracies = [self.expected_accuracy] * len(epochs)
        self.ax.plot(epochs, expected_accuracies, color='red', linestyle='--', linewidth=2, label='Hypothesis Line')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        plt.ioff()  # Turn off interactive mode
        plt.show()

# Define ResNet10 architecture
def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = -1  # For 'channels_last' data format
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

def ResNet10(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial layers
    x = Conv2D(32, 7, strides=2, padding='same', kernel_regularizer=l2(l2_weight_conv), name='conv1_conv')(inputs)
    x = BatchNormalization(axis=-1, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # ResNet blocks
    x = resnet_block(x, 32, conv_shortcut=True, name='conv2_block1')
    x = resnet_block(x, 32, conv_shortcut=False, name='conv2_block2')

    x = resnet_block(x, 64, stride=2, conv_shortcut=True, name='conv3_block1')
    x = resnet_block(x, 64, conv_shortcut=False, name='conv3_block2')

    x = resnet_block(x, 128, stride=2, conv_shortcut=True, name='conv4_block1')
    x = resnet_block(x, 128, conv_shortcut=False, name='conv4_block2')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5)(x)  # Add dropout layer

    # Increase the number of dense layers for better classification
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_weight_dense), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_weight_dense), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_weight_dense), name='fc3')(x)

    model = Model(inputs, x, name='resnet10_2d')
    return model

# Initialize the model once before the cross-validation loop
model_2d = ResNet10(input_shape, num_classes)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model_2d.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Save the initial weights
initial_weights = model_2d.get_weights()

# Main K-Fold Cross Validation
expected_accuracy = 0.8  # Hypothetical expected accuracy for the hypothesis line
fold_no = 1
for train_paths, train_labels, val_paths, val_labels in tqdm(stratified_kfold(all_train_folder_paths, all_train_folder_labels, n_splits=5), desc="K-Fold Progress"):
    print(f'\nTraining fold {fold_no}...')

    # Count folders per class
    train_count_per_class = {gesture_name: 0 for gesture_name in gesture_names}
    val_count_per_class = {gesture_name: 0 for gesture_name in gesture_names}

    for path, label in zip(train_paths, train_labels):
        train_count_per_class[gesture_names[label]] += 1

    for path, label in zip(val_paths, val_labels):
        val_count_per_class[gesture_names[label]] += 1

    print(f'Fold {fold_no} - Training set folder count per class:')
    for gesture_name in gesture_names:
        print(f'{gesture_name}: {train_count_per_class[gesture_name]}')

    print(f'Fold {fold_no} - Validation set folder count per class:')
    for gesture_name in gesture_names:
        print(f'{gesture_name}: {val_count_per_class[gesture_name]}')

    # Logging the exact folder names used in each fold
    print(f'Fold {fold_no} - Training set folder names:')
    for path in train_paths:
        print(path)

    print(f'Fold {fold_no} - Validation set folder names:')
    for path in val_paths:
        print(path)

    train_photo_paths, train_labels_processed = [], []
    val_photo_paths, val_labels_processed = [], []

    for folder, label in zip(train_paths, train_labels):
        paths, labels = load_photo_paths_from_folders([folder], label)
        train_photo_paths.extend(paths)
        train_labels_processed.extend(labels)

    for folder, label in zip(val_paths, val_labels):
        paths, labels = load_photo_paths_from_folders([folder], label)
        val_photo_paths.extend(paths)
        val_labels_processed.extend(labels)

    train_fold_dataset = create_dataset(train_photo_paths, train_labels_processed, batch_size)
    val_fold_dataset = create_dataset(val_photo_paths, val_labels_processed, batch_size)

    # Calculate steps per epoch and validation steps
    steps_per_epoch = len(train_photo_paths) // batch_size
    validation_steps = len(val_photo_paths) // batch_size

    # Restore model weights from the previous fold (if not the first fold)
    if fold_no > 1:
        model_2d.set_weights(previous_weights)

    learning_rate_logger = LearningRateLogger()

    # Train the model using the data generators
    history_2d = model_2d.fit(
        train_fold_dataset,
        validation_data=val_fold_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=15,
        initial_epoch=0,
        callbacks=[
            checkpoint_callback,
            learning_rate_logger,
            tensorboard_callback,
            RealTimeGraph(expected_accuracy),
            MainValidationLogger(main_val_data=main_val_dataset, main_validation_steps=main_validation_steps)
        ]
    )

    # Evaluate the model on the validation set of the current fold
    val_loss, val_accuracy = model_2d.evaluate(val_fold_dataset, steps=validation_steps)

    # Store metrics for the current fold
    fold_val_losses.append(val_loss)
    fold_val_accuracies.append(val_accuracy)

    # Save model weights to use in the next fold
    previous_weights = model_2d.get_weights()

    print(f'Finished fold {fold_no} with val_loss={val_loss}, val_accuracy={val_accuracy}')
    fold_no += 1
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Add
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import time

# Set matplotlib backend to 'Agg'
matplotlib_use('Agg')

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Local paths
local_train_dir = '/Users/esmaelmoataz/Documents/Machine learning/Hand gesture/train'
local_val_dir = '/Users/esmaelmoataz/Documents/Machine learning/Hand gesture/val'

# Gesture names
gesture_names = ['dislike', 'like', 'stop']  # Add other gesture names as needed
num_classes = len(gesture_names)

# Function to preprocess individual photos
def preprocess_photo(photo_path):
    img_string = tf.io.read_file(photo_path)
    img = tf.image.decode_image(img_string, channels=1)
    img.set_shape([None, None, 1])  # Set shape after decoding to include the channel dimension
    img = tf.image.resize(img, (128, 128))  # Resize image
    img.set_shape([128, 128, 1])  # Explicitly set the final shape after resizing
    img = tf.cast(img, tf.float32) / 255.0
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
input_shape = (128, 128, 1)  # Updated to match new image size
batch_size = 32  # Adjust batch size as needed

# Regularization parameters
l2_weight_conv = 0
l2_weight_dense = 0

# Extract folder paths and labels
all_train_folder_paths = [fp[0] for fp in train_folder_paths]
all_train_folder_labels = [gesture_names.index(fp[1]) for fp in train_folder_paths]

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

# Custom function to handle stratified K-Fold
def stratified_kfold(all_folder_paths, all_folder_labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(all_folder_paths, all_folder_labels):
        train_paths = [all_folder_paths[i] for i in train_index]
        train_labels = [all_folder_labels[i] for i in train_index]
        val_paths = [all_folder_paths[i] for i in val_index]
        val_labels = [all_folder_labels[i] for i in val_index]
        yield train_paths, train_labels, val_paths, val_labels

# Initialize lists to store metrics for each fold
fold_val_accuracies = []
fold_val_losses = []

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

# Custom callback for real-time graph
class RealTimeGraph(Callback):
    def __init__(self, expected_accuracy):
        super().__init__()
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.colors = plt.cm.viridis(np.linspace(0, 1, 20))  # Use a colormap with 20 different colors
        self.epoch_data = []
        self.expected_accuracy = expected_accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Store epoch data
        self.epoch_data.append((epoch, logs['val_accuracy']))
        self.ax.clear()
        self.ax.set_xlim(0, len(self.epoch_data) + 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Validation Accuracy')

        # Plot data points
        epochs = [e for e, _ in self.epoch_data]
        accuracies = [acc for _, acc in self.epoch_data]
        self.ax.plot(epochs, accuracies, color='blue', linestyle='-', linewidth=2, label='Actual Accuracy')

        # Plot hypothesis line
        expected_accuracies = [self.expected_accuracy] * len(epochs)
        self.ax.plot(epochs, expected_accuracies, color='red', linestyle='--', linewidth=2, label='Hypothesis Line')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        plt.ioff()  # Turn off interactive mode
        plt.show()

# Define ResNet10 architecture
def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = -1  # For 'channels_last' data format
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

def ResNet10(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial layers
    x = Conv2D(32, 7, strides=2, padding='same', kernel_regularizer=l2(l2_weight_conv), name='conv1_conv')(inputs)
    x = BatchNormalization(axis=-1, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # ResNet blocks
    x = resnet_block(x, 32, conv_shortcut=True, name='conv2_block1')
    x = resnet_block(x, 32, conv_shortcut=False, name='conv2_block2')

    x = resnet_block(x, 64, stride=2, conv_shortcut=True, name='conv3_block1')
    x = resnet_block(x, 64, conv_shortcut=False, name='conv3_block2')

    x = resnet_block(x, 128, stride=2, conv_shortcut=True, name='conv4_block1')
    x = resnet_block(x, 128, conv_shortcut=False, name='conv4_block2')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5)(x)  # Add dropout layer

    # Increase the number of dense layers for better classification
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_weight_dense), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_weight_dense), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_weight_dense), name='fc3')(x)

    model = Model(inputs, x, name='resnet10_2d')
    return model

# Initialize the model once before the cross-validation loop
model_2d = ResNet10(input_shape, num_classes)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model_2d.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Save the initial weights
initial_weights = model_2d.get_weights()

# Main K-Fold Cross Validation
expected_accuracy = 0.8  # Hypothetical expected accuracy for the hypothesis line
fold_no = 1
for train_paths, train_labels, val_paths, val_labels in tqdm(stratified_kfold(all_train_folder_paths, all_train_folder_labels, n_splits=5), desc="K-Fold Progress"):
    print(f'\nTraining fold {fold_no}...')

    # Count folders per class
    train_count_per_class = {gesture_name: 0 for gesture_name in gesture_names}
    val_count_per_class = {gesture_name: 0 for gesture_name in gesture_names}

    for path, label in zip(train_paths, train_labels):
        train_count_per_class[gesture_names[label]] += 1

    for path, label in zip(val_paths, val_labels):
        val_count_per_class[gesture_names[label]] += 1

    print(f'Fold {fold_no} - Training set folder count per class:')
    for gesture_name in gesture_names:
        print(f'{gesture_name}: {train_count_per_class[gesture_name]}')

    print(f'Fold {fold_no} - Validation set folder count per class:')
    for gesture_name in gesture_names:
        print(f'{gesture_name}: {val_count_per_class[gesture_name]}')

    # Logging the exact folder names used in each fold
    print(f'Fold {fold_no} - Training set folder names:')
    for path in train_paths:
        print(path)

    print(f'Fold {fold_no} - Validation set folder names:')
    for path in val_paths:
        print(path)

    train_photo_paths, train_labels_processed = [], []
    val_photo_paths, val_labels_processed = [], []

    for folder, label in zip(train_paths, train_labels):
        paths, labels = load_photo_paths_from_folders([folder], label)
        train_photo_paths.extend(paths)
        train_labels_processed.extend(labels)

    for folder, label in zip(val_paths, val_labels):
        paths, labels = load_photo_paths_from_folders([folder], label)
        val_photo_paths.extend(paths)
        val_labels_processed.extend(labels)

    train_fold_dataset = create_dataset(train_photo_paths, train_labels_processed, batch_size)
    val_fold_dataset = create_dataset(val_photo_paths, val_labels_processed, batch_size)

    # Calculate steps per epoch and validation steps
    steps_per_epoch = len(train_photo_paths) // batch_size
    validation_steps = len(val_photo_paths) // batch_size

    # Restore model weights from the previous fold (if not the first fold)
    if fold_no > 1:
        model_2d.set_weights(previous_weights)

    learning_rate_logger = LearningRateLogger()

    # Train the model using the data generators
    history_2d = model_2d.fit(
        train_fold_dataset,
        validation_data=val_fold_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=15,
        initial_epoch=0,
        callbacks=[
            checkpoint_callback,
            learning_rate_logger,
            tensorboard_callback,
            RealTimeGraph(expected_accuracy),
            MainValidationLogger(main_val_data=main_val_dataset, main_validation_steps=main_validation_steps)
        ]
    )

    # Evaluate the model on the validation set of the current fold
    val_loss, val_accuracy = model_2d.evaluate(val_fold_dataset, steps=validation_steps)

    # Store metrics for the current fold
    fold_val_losses.append(val_loss)
    fold_val_accuracies.append(val_accuracy)

    # Save model weights to use in the next fold
    previous_weights = model_2d.get_weights()

    print(f'Finished fold {fold_no} with val_loss={val_loss}, val_accuracy={val_accuracy}')
    fold_no += 1

# Calculate average metrics across all folds
average_val_loss = np.mean(fold_val_losses)
average_val_accuracy = np.mean(fold_val_accuracies)
print(f'Average Validation Loss: {average_val_loss}, Average Validation Accuracy: {average_val_accuracy}')

# Final evaluation on the independent validation set
final_val_loss, final_val_accuracy = model_2d.evaluate(main_val_dataset, steps=main_validation_steps)
print(f'Final validation set performance - Loss: {final_val_loss}, Accuracy: {final_val_accuracy}')

# Calculate average metrics across all folds
average_val_loss = np.mean(fold_val_losses)
average_val_accuracy = np.mean(fold_val_accuracies)
print(f'Average Validation Loss: {average_val_loss}, Average Validation Accuracy: {average_val_accuracy}')

# Final evaluation on the independent validation set
final_val_loss, final_val_accuracy = model_2d.evaluate(main_val_dataset, steps=main_validation_steps)
print(f'Final validation set performance - Loss: {final_val_loss}, Accuracy: {final_val_accuracy}')
