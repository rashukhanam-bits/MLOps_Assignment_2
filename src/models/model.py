"""Baseline CNN for 224x224 RGB binary classification (cat=0, dog=1)."""
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation_layer():
    """Applied only during training, directly inside the model graph."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_model(image_size: int = 224) -> keras.Model:
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = build_augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="cats_dogs_cnn")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
