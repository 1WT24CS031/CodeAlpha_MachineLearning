import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load EMNIST Balanced dataset
# --------------------------------------------------

print("Loading EMNIST Balanced dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = ds_info.features["label"].num_classes

print("Dataset loaded successfully!")
print("Training examples:", ds_info.splits["train"].num_examples)
print("Testing examples:", ds_info.splits["test"].num_examples)
print("Number of classes:", NUM_CLASSES)


# --------------------------------------------------
# 2. Preprocess images
# --------------------------------------------------

def preprocess(image, label):
    # Convert pixel values from 0-255 to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    # Add grayscale channel
    image = tf.expand_dims(image, axis=-1)

    return image, label


ds_train = ds_train.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

ds_test = ds_test.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# 3. Create efficient data pipelines
# --------------------------------------------------

BATCH_SIZE = 128

ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# 4. Build CNN model
# --------------------------------------------------

print("\nBuilding CNN model...")

model = tf.keras.Sequential([
    
    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps to a single vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# --------------------------------------------------
# 5. Compile the CNN
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 6. Display model architecture
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 7. Save sample images
# --------------------------------------------------

images, labels = next(iter(ds_train))

plt.figure(figsize=(10, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap="gray")
    plt.title(f"Class: {labels[i].numpy()}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/sample_characters.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nSample image visualization saved successfully!")
print("Image batch shape:", images.shape)

print("\nCNN model created successfully!")
# --------------------------------------------------
# 8. Train the CNN
# --------------------------------------------------

print("\nStarting CNN training...")

EPOCHS = 5

history = model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test
)

print("\nCNN training completed successfully!")
# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

print("\nEvaluating CNN model...")

test_loss, test_accuracy = model.evaluate(ds_test, verbose=1)

print("\nTest Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)
# --------------------------------------------------
# 10. Plot training history
# --------------------------------------------------

print("\nCreating training graphs...")

# Accuracy graph
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/training_accuracy.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# Loss graph
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/training_loss.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Training graphs saved successfully!")
# --------------------------------------------------
# 11. Generate predictions
# --------------------------------------------------

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images_batch, labels_batch in ds_test:
    predictions = model.predict(images_batch, verbose=0)
    
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels_batch.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# --------------------------------------------------
# 12. Classification report
# --------------------------------------------------

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    zero_division=0
)

print(report)


# --------------------------------------------------
# 13. Confusion matrix
# --------------------------------------------------

print("Creating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("EMNIST Character Recognition - Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved successfully!")
# --------------------------------------------------
# 14. Save trained CNN model
# --------------------------------------------------

print("\nSaving trained CNN model...")

model.save("models/handwritten_character_cnn.keras")

print("CNN model saved successfully!")
print("Project completed successfully!")
# --------------------------------------------------
# 15. Visualize model predictions
# --------------------------------------------------

print("\nCreating prediction visualization...")

images, labels = next(iter(ds_test))

predictions = model.predict(images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    plt.imshow(
        images[i].numpy().squeeze(),
        cmap="gray"
    )

    plt.title(
        f"True: {labels[i].numpy()} | Pred: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/prediction_examples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Prediction visualization saved successfully!")