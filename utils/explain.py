import numpy as np
import matplotlib.pyplot as plt
import shap
import cv2
import tensorflow as tf

# ----------------------------------
# 1. FEATURE IMPORTANCE (TEXT MODEL)
# ----------------------------------

def plot_feature_importance(model, feature_names):
    """
    Works for RandomForest / Tree models
    """

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()

        plt.savefig("static/feature_importance.png")
        plt.close()
    else:
        print("Model does not support feature_importances_")


# ----------------------------------
# 2. SHAP EXPLANATION (ADVANCED)
# ----------------------------------

def shap_explanation(model, X_sample, feature_names):
    """
    SHAP explanation for any ML model
    """

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)

    plt.savefig("static/shap_plot.png", bbox_inches='tight')
    plt.close()


# ----------------------------------
# 3. GRAD-CAM (IMAGE MODEL)
# ----------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap.numpy()


def save_gradcam(img_path, model, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img

    output_path = "static/gradcam.jpg"
    cv2.imwrite(output_path, superimposed_img)

    return output_path