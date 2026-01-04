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

FILE_ID = "1MBqyS3opfYxVslAoNJM5rlHShOYOF2hU"
model = None


# -----------------  SAFE MODEL LOADER  -----------------
def load_model_safe():
    global model
    if model is not None:
        return model

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # download if not there
    if not os.path.exists(MODEL_PATH):
        print(">>> Downloading model...")
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # verify size (Railway sometimes corrupts file)
    if os.path.getsize(MODEL_PATH) < 5000000:
        print(">>> CORRUPTED MODEL - Re-downloading...")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    print(">>> Loading TensorFlow model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # warmup build
    model(tf.zeros((1,224,224,3)))
    print(">>> Model Ready")

    return model


class_names = ['glioma','meningioma','notumor','pituitary']



# -----------------  GRAD-CAM  -----------------
def get_gradcam(img_array, mdl):
    try:
        target_layer = mdl.get_layer("conv2d_2")   # ← your real last conv
    except:
        # fallback in case TF renames it
        for layer in mdl.layers[::-1]:
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break

    grad_model = tf.keras.models.Model(
        inputs=mdl.input,
        outputs=[target_layer.output, mdl.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_out)[0]
    conv_out = conv_out[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_out.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-10)
    cam = cv2.resize(cam, (224, 224))

    return cam




# -----------------  MAIN VIEW  -----------------
def predict(request):
    result = ""
    file_url = ""
    heatmap_url = ""
    message = ""
    box_class = ""
    prob = ""
    model_acc = 95.6

    if request.method == "POST" and request.FILES.get("image"):
        mdl = load_model_safe()

        img = request.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        file_path = fs.path(filename)
        file_url = fs.url(filename)

        img_data = image.load_img(file_path, target_size=(224,224))
        x = image.img_to_array(img_data) / 255.0
        x = np.expand_dims(x, axis=0)

        prediction = mdl.predict(x)
        class_index = np.argmax(prediction)
        result = class_names[class_index]

        confidence = float(np.max(prediction)) * 100
        prob = round(confidence, 2)

        if result.lower() == "notumor":
            message = "No Tumor Detected."
            box_class = "safe"
        else:
            message = f"Tumor Detected — {result.capitalize()}"
            box_class = "danger"

        heatmap = get_gradcam(x, model)

        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        final = heatmap * 0.5 + img_cv

        heatmap_path = fs.path("heatmap_" + filename)
        cv2.imwrite(heatmap_path, final)
        heatmap_url = fs.url("heatmap_" + filename)

    return render(request,"index.html",{
        "result": result,
        "message": message,
        "prob": prob,
        "model_acc": model_acc,
        "file_url": file_url,
        "heatmap_url": heatmap_url,
        "box_class": box_class
    })
