from django.shortcuts import render
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from django.core.files.storage import FileSystemStorage
import os
import gdown


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model_multi.h5")

DRIVE_MODEL_URL = "https://drive.google.com/uc?id=1MBqyS3opfYxVslAoNJM5rlHShOYOF2hU"

if not os.path.exists(MODEL_PATH):
    print(">>> Downloading model from Drive...")
    gdown.download(DRIVE_MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# IMPORTANT: FORCE BUILD MODEL
model.build((None, 224, 224, 3))
model.predict(np.zeros((1,224,224,3)))

class_names = ['glioma','meningioma','notumor','pituitary']


# -------- GRAD CAM --------
def get_gradcam(img_array):
    # ensure model has graph
    model.predict(img_array)

    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=model.layers[0].input,   # 👈 FIXED HERE
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0,1))

    cam = tf.zeros(conv_outputs.shape[0:2])

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:,:,i]

    cam = tf.maximum(cam,0)
    cam = cam / tf.reduce_max(cam)
    heatmap = cv2.resize(cam.numpy(), (224,224))

    return heatmap
