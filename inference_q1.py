# inference_q1.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Replace or add image paths here
test_images = [
    "Q1/test/cat/Cat (1).jpg",
    "Q1/test/cat/Cat (2).jpg",
    "Q1/test/dog/Dog (1).jpg",
    "Q1/test/dog/Dog (2).jpg"
]

for img_path in test_images:
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)[0][0]
        label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"

        print(f"{img_path} → {label}")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
