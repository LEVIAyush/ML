import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

# Set Parameters
IMG_SIZE = 224
BATCH_SIZE = 24  # Customized batch size to make it less standard
EPOCHS = 15  # Adjusted epoch count for specific use case
DATA_DIR = "C:/Users/levia/Downloads/xry/Covid19-dataset"

# Helper function to create ImageDataGenerators
def create_generators(data_dir, img_size, batch_size):
    """Sets up training, validation, and test data generators."""
    train_data = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.25,  # Adjusted validation split
        rotation_range=25,  # Slightly higher augmentation
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    test_data = ImageDataGenerator(rescale=1.0 / 255)
    
    train_generator = train_data.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
    )
    validation_generator = train_data.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
    )
    test_generator = test_data.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )
    return train_generator, validation_generator, test_generator

# Data Preparation
train_gen, val_gen, test_gen = create_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

# Calculate Class Weights
classes = np.unique(train_gen.classes)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_gen.classes,
)
class_weights = {i: weights[i] for i in range(len(weights))}

# Load and Customize Pre-trained Models
def create_hybrid_model(input_shape):
    """Builds a hybrid model combining VGG19 and EfficientNetB0."""
    vgg19_base = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    efficientnet_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Freeze base layers to retain pre-trained weights
    vgg19_base.trainable = False
    efficientnet_base.trainable = False
    
    # Process inputs through both models
    input_tensor = Input(shape=input_shape)
    vgg19_output = GlobalAveragePooling2D()(vgg19_base(input_tensor))
    efficientnet_output = GlobalAveragePooling2D()(efficientnet_base(input_tensor))
    
    # Concatenate and build dense layers
    merged = Concatenate()([vgg19_output, efficientnet_output])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.4)(x)  # Dropout rate tuned for potential overfitting
    output = Dense(1, activation="sigmoid")(x)
    return Model(inputs=input_tensor, outputs=output)

# Compile Model
model = create_hybrid_model((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define Callbacks
lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

# Train Model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[lr_reduction, early_stopping],
)

# Save Model and Training History
model.save("C:/Users/levia/OneDrive/Desktop/ML/Covid_hybrid_v2.h5")
with open("C:/Users/levia/OneDrive/Desktop/ML/training_history_v2.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluate Model
test_gen.reset()
predictions = (model.predict(test_gen) > 0.5).astype("int32")

# Metrics and Visualization
print(classification_report(test_gen.classes, predictions, target_names=["Non-COVID", "COVID"]))
auc = roc_auc_score(test_gen.classes, model.predict(test_gen))
print(f"AUC Score: {auc:.2f}")
cm = confusion_matrix(test_gen.classes, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-COVID", "COVID"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
