from django.shortcuts import render
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from django.core.files.storage import FileSystemStorage
import os

import gdown


# MODEL DOWNLOAD CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model_multi.h5")

DRIVE_MODEL_URL = "https://drive.google.com/uc?id=1MBqyS3opfYxVslAoNJM5rlHShOYOF2hU"

# Download model if not already there
if not os.path.exists(MODEL_PATH):
    print(">>> Downloading model from Drive...")
    gdown.download(DRIVE_MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warmup
model(tf.zeros((1,224,224,3)))

class_names = ['glioma','meningioma','notumor','pituitary']


# -------- GRAD CAM --------
def get_gradcam(img_array):
    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.zeros(conv_outputs.shape[0:2])

    for i,w in enumerate(weights):
        cam += w * conv_outputs[:,:,i]

    cam = tf.maximum(cam,0)
    cam = cam / tf.reduce_max(cam)
    heatmap = cv2.resize(cam.numpy(), (224,224))

    return heatmap


# -------- MAIN VIEW --------
def predict(request):
    result = ""
    file_url = ""
    heatmap_url = ""
    message = ""
    box_class = ""
    prob = ""
    model_acc = 95.6   # set your trained accuracy

    if request.method == "POST" and request.FILES.get("image"):
        img = request.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        file_path = fs.path(filename)
        file_url = fs.url(filename)

        img_data = image.load_img(file_path, target_size=(224,224))
        x = image.img_to_array(img_data) / 255.0
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x)
        class_index = np.argmax(prediction)
        result = class_names[class_index]

        confidence = float(np.max(prediction)) * 100
        prob = round(confidence, 2)

        if result.lower() == "notumor":
            message = "No Tumor Detected. Stay healthy and keep taking care of yourself!"
            box_class = "safe"
        else:
            message = f"Tumor Detected — {result.capitalize()} ⚠ Please consult a medical professional."
            box_class = "danger"

        # heatmap
        heatmap = get_gradcam(x)
        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv, (224,224))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        final = heatmap * 0.5 + img_cv

        heatmap_path = fs.path("heatmap_" + filename)
        cv2.imwrite(heatmap_path, final)
        heatmap_url = fs.url("heatmap_" + filename)

    return render(request, "index.html",{
        "result": result,
        "message": message,
        "prob": prob,
        "model_acc": model_acc,
        "file_url": file_url,
        "heatmap_url": heatmap_url,
        "box_class": box_class
    })
