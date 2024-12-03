import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Average
from tensorflow.keras.applications import VGG19, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Parameters
img_size = 224
batch_size = 32
datadir = "C:/Users/levia/Downloads/xry/Covid19-dataset"
epochs = 15

# Data Generators
train_data = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = train_data.flow_from_directory(
    os.path.join(datadir, 'train'),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
)

validation_generator = train_data.flow_from_directory(
    os.path.join(datadir, 'train'),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
)

test_data = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_data.flow_from_directory(
    os.path.join(datadir, 'test'),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
)

# Compute Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Load Pre-trained Models
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the layers
vgg19_base.trainable = False
efficientnet_base.trainable = False

# Define Inputs
input_vgg = Input(shape=(img_size, img_size, 3))
input_efficientnet = Input(shape=(img_size, img_size, 3))

# Get Outputs from Both Models
vgg19_output = vgg19_base(input_vgg)
efficientnet_output = efficientnet_base(input_efficientnet)

# Flatten the Outputs
vgg19_flattened = Flatten()(vgg19_output)
efficientnet_flattened = Flatten()(efficientnet_output)

# Add Dense Layers to Match Shapes
vgg19_dense = Dense(128, activation='relu')(vgg19_flattened)
efficientnet_dense = Dense(128, activation='relu')(efficientnet_flattened)

# Combine the Outputs
combined_output = Average()([vgg19_dense, efficientnet_dense])

# Add a Final Classification Layer
final_output = Dense(1, activation='sigmoid')(combined_output)

# Create the Ensemble Model
model = Model(inputs=[input_vgg, input_efficientnet], outputs=final_output)

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


def dual_generator_with_weights(generator, class_weights):
    """
    Wraps a generator to include sample weights for each batch.
    Arguments:
        generator: The data generator yielding batches of (x, y).
        class_weights: A dictionary mapping class indices to weights.
    Yields:
        A tuple: ([input_1, input_2], y, sample_weights).
    """
    while True:
        x, y = next(generator)  # Get the next batch of data
        # Compute sample weights for each sample in the batch
        sample_weights = np.array([class_weights[int(label)] for label in y])

        # Yield dual inputs, labels, and sample weights
        yield ([x, x], y, sample_weights)

# Wrap the generator in a tf.data.Dataset
def create_tf_dataset(generator, class_weights):
    output_signature = (
        (tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32)),  # For dual inputs
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Labels
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Sample weights
    )

    return tf.data.Dataset.from_generator(
        lambda: dual_generator_with_weights(generator, class_weights),
        output_signature=output_signature
    )

# Create tf.data.Dataset objects for training and validation
train_dataset = create_tf_dataset(train_generator, class_weights)
validation_dataset = create_tf_dataset(validation_generator, class_weights)

# Model training
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[lr_reduction, early_stopping],
    verbose=1,  # Ensures clear progress display
)





# Save Model and Training History
model.save("C:/Users/levia/OneDrive/Desktop/ML/Covid.h5")
with open("C:/Users/levia/OneDrive/Desktop/ML/training_history.pkl", "wb") as f:
    import pickle
    pickle.dump(history.history, f)

# Evaluate Model
test_generator.reset()
predictions = (model.predict([test_generator, test_generator]) > 0.5).astype("int32")

# Classification Report
print(classification_report(test_generator.classes, predictions, target_names=["Non-COVID", "COVID"]))

# AUC Score
auc = roc_auc_score(test_generator.classes, model.predict([test_generator, test_generator]))
print(f"AUC Score: {auc:.2f}")

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-COVID", "COVID"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()