import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_image

def validate_labels_json(file_path):
    valid_data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and "image_path" in data and "labels" in data:
                        valid_data.append(data)
                except json.JSONDecodeError:
                    print(f"Advertencia: Línea inválida en '{file_path}'.")
        if valid_data:
            with open(file_path, "w") as f:
                for item in valid_data:
                    json.dump(item, f)
                    f.write("\n")
    return valid_data

def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(80, 80, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    return model

def train_model(data_dir="data", confirmed_dir="confirmed_data", epochs=5):
    if not os.path.exists("classes.txt"):
        print("Error: Falta 'classes.txt'. Crea este archivo con las clases detectables.")
        return None

    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    all_data = []
    for directory in [data_dir, confirmed_dir]:
        labels_path = f"{directory}/labels.json"
        all_data.extend(validate_labels_json(labels_path))

    if not all_data:
        print("No hay datos de entrenamiento disponibles.")
        return None

    images, labels = [], []
    for item in all_data:
        try:
            img = load_image(item["image_path"])
            images.append(img)
            label_vector = [1 if cls in item["labels"] else 0 for cls in classes]
            labels.append(label_vector)
        except FileNotFoundError:
            print(f"Imagen no encontrada: {item['image_path']}.")

    if not images:
        print("No se encontraron imágenes válidas para entrenar.")
        return None

    images = np.array(images)
    labels = np.array(labels)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    try:
        model = tf.keras.models.load_model("models/model.h5")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        print("Modelo cargado y compilado desde 'models/model.h5'")
    except:
        print("No se pudo cargar el modelo existente. Creando uno nuevo...")
        model = build_model(len(classes))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss="binary_crossentropy", metrics=["accuracy"])

    train_gen = datagen.flow(images, labels, batch_size=16)
    model.fit(train_gen, epochs=epochs)
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    print("Modelo entrenado y guardado en 'models/model.h5'")
    return model

def load_or_train_model():
    try:
        model = tf.keras.models.load_model("models/model.h5")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        print("Modelo cargado y compilado desde 'models/model.h5'")
        return model
    except:
        print("Error al cargar el modelo. Entrenando uno nuevo...")
        return train_model()

def predict(model, image):
    if model is None:
        return [0] * 3
    image = tf.image.resize(image, (80, 80))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return model.predict(image, verbose=0)[0]