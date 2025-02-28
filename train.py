import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import tensorflow as tf

# Define paths
dataset_path = "/Users/pranav/Desktop/Train/"
real = "/Users/pranav/Desktop/Train/Real/"
fake = "/Users/pranav/Desktop/Train/Fake/"

# Visualizing real and fake faces
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (96, 96))
    return image[..., ::-1]  # Convert BGR to RGB

# Ensure enough images exist
real_images = os.listdir(real)
fake_images = os.listdir(fake)
if len(real_images) < 16 or len(fake_images) < 16:
    raise ValueError("Ensure at least 16 images in both Real and Fake directories.")

# Plot real faces
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(real, real_images[i])))
    plt.suptitle("Real Faces", fontsize=20)
    plt.axis('off')
plt.show()

# Plot fake faces
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(fake, fake_images[i])))
    plt.suptitle("Fake Faces", fontsize=20)
    plt.axis('off')
plt.show()

# Data generator with no augmentation
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = data_generator.flow_from_directory(dataset_path,
                                           class_mode="binary",
                                           target_size=(96, 96),
                                           batch_size=32,
                                           subset="training")
val = data_generator.flow_from_directory(dataset_path,
                                         class_mode="binary",
                                         target_size=(96, 96),
                                         batch_size=32,
                                         subset="validation", shuffle=False)

# MobileNetV2 base model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
mnet.trainable = True
for layer in mnet.layers[:100]:
    layer.trainable = False

# Model architecture
model = Sequential([
    mnet,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# Compile the model
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# Learning rate scheduler
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_scheduler = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
hist = model.fit(train,
                 epochs=4,  # Set epochs to 4
                 validation_data=val,
                 callbacks=[lr_scheduler, reduce_lr],
                 verbose=1)

# Save the model
model.save('/Users/pranav/Desktop/deepfake_detection.h5')

# Visualizing accuracy and loss
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(len(train_loss))  # Adjust to the correct number of epochs

# Plot Loss
plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss, label='Train Loss')
plt.plot(xc, val_loss, label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend()
plt.style.use(['classic'])
plt.show()

# Plot Accuracy
plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc, label='Train Accuracy')
plt.plot(xc, val_acc, label='Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(loc='lower right')
plt.style.use(['classic'])
plt.show()

# Evaluate the model
# Reinitialize the ImageDataGenerator for evaluation without augmentation
data_no_aug = ImageDataGenerator(rescale=1./255, validation_split=0.5)
val = data_no_aug.flow_from_directory(dataset_path,
                                      class_mode="binary",
                                      target_size=(96, 96),
                                      batch_size=32,
                                      subset="validation",
                                      shuffle=False)

# Load the saved model
model = tf.keras.models.load_model('/Users/pranav/Desktop/deepfake_detection.h5')

# Get true labels from the validation generator
y_true = val.classes  # True labels from the validation set

# Predict on the validation dataset
y_pred_prob = model.predict(val)  # Predicted probabilities
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)  # Convert probabilities to binary class labels

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
print("\nClassification Report:\n", report)

# Compute F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for imbalanced datasets
print("\nF1 Score (Weighted):", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)
