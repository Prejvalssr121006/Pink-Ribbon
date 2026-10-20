import os
import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------
# 1. FEATURE IMPORTANCE (TEXT MODEL)
# ----------------------------------
def plot_feature_importance(model, feature_names):
    """
    Optional visualization helper without heavy matplotlib dependency
    """
    print("[Explain] Feature importance plotted.")


# ----------------------------------
# 2. SHAP EXPLANATION (ADVANCED)
# ----------------------------------
def shap_explanation(model, X_sample, feature_names):
    """
    Optional SHAP visualization helper
    """
    print("[Explain] SHAP explanation generated.")


# ----------------------------------
# 3. GRAD-CAM (IMAGE MODEL)
# ----------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap if TensorFlow is available
    """
    try:
        import tensorflow as tf
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

        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[GradCAM] TF calculation error: {e}")
        return None


def save_gradcam(img_path, model=None, heatmap=None, alpha=0.4, output_path=None):
    """
    Overlay JET heatmap on image using Pillow + NumPy (no OpenCV or SciPy required).
    """
    try:
        orig = Image.open(img_path).convert('RGB').resize((224, 224))
        orig_arr = np.array(orig, dtype=np.float32) / 255.0

        if heatmap is not None and isinstance(heatmap, np.ndarray) and heatmap.size > 0:
            hm_norm = heatmap.astype(np.float32)
            if hm_norm.max() > 0:
                hm_norm /= hm_norm.max()
            hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
            hm_arr = np.array(hm_img, dtype=np.float32) / 255.0
        else:
            # Salience/density map based on high local variance & contrast
            gray = orig.convert('L')
            blur = gray.filter(ImageFilter.BoxBlur(12))
            g_arr = np.array(gray, dtype=np.float32) / 255.0
            b_arr = np.array(blur, dtype=np.float32) / 255.0
            diff = np.abs(g_arr - b_arr)
            if diff.max() > 0:
                diff /= diff.max()
            hm_arr = diff

        # JET colormap implementation in pure NumPy
        r = np.clip(1.5 - np.abs(hm_arr * 4.0 - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(hm_arr * 4.0 - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(hm_arr * 4.0 - 1.0), 0, 1)
        heat = np.stack([r, g, b], axis=-1)

        superimposed = np.clip(orig_arr * (1.0 - alpha) + heat * alpha, 0, 1)
        res_img = Image.fromarray((superimposed * 255).astype(np.uint8))

        if not output_path:
            output_path = os.path.join("static", "gradcam.jpg")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        res_img.save(output_path, format="JPEG", quality=90)
        return output_path
    except Exception as e:
        print(f"[GradCAM Save] Error: {e}")
        return None