from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import numpy as np
import matplotlib
from PIL import Image
import keras

import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries


app = Flask(__name__)


dic = {0: 'Bacterial Pneumonia', 1: 'COVID-19', 2: 'Normal', 3: 'Tuberculosis', 4: 'Viral Pneumonia'}

model = load_model('Multiclass_InceptionV3.h5')  # Load InceptionV3 model
img_size = (256, 256)
last_conv_layer_name = "conv2d_93"  # Change this if needed for InceptionV3

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Your Grad-CAM code here...
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="static/cam_image.jpg", alpha=0.4):
    # Your Grad-CAM code here...
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(256, 256))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 256, 256, 3)
    p = model.predict(i)
    predicted_class_index = np.argmax(p, axis=1)[0]
    return dic[predicted_class_index], p

@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        prediction, _ = predict_label(img_path)
        return render_template("index.html", prediction=prediction, img_path=img_path)
    return render_template("index.html", prediction=None, img_path=None)

@app.route("/about")
def about_page():
    return "Please subscribe to Artificial Intelligence Hub..!!!"

@app.route("/lime", methods=['GET'])
def lime_explanation():
    img_path = request.args.get('img_path')

    # Load original image (for size)
    original_img = Image.open(img_path)
    original_size = original_img.size  # (width, height)

    # Model input
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0],
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # Create LIME visualization
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    lime_img = Image.fromarray(np.uint8(lime_img * 255))

    # 🔥 Resize to original image size
    lime_img = lime_img.resize(original_size, Image.BILINEAR)

    lime_path = "static/lime_image.jpg"
    lime_img.save(lime_path)

    return jsonify({'lime_path': lime_path})


@app.route("/grad_cam", methods=['GET'])
def grad_cam():
    img_path = request.args.get('img_path')
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    cam_path = "static/cam_image.jpg"
    save_and_display_gradcam(img_path, heatmap, cam_path)
    return jsonify({'cam_path': cam_path})

if __name__ == '__main__':
    app.run(debug=True)
