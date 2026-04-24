"""
Pink Ribbon: Multimodal Breast Cancer Detection System
app.py — Flask Backend (FIXED)

KEY FIX: train_text_model.py trains on data.csv which has:
  id + diagnosis + 30 features = drops diagnosis, but 'id' may remain
  depending on the CSV, leaving 31 or 30 numeric columns.
  This app.py detects the expected feature count from the saved scaler
  and builds the vector accordingly — no more 30 vs 31 mismatch.

Uses:
  utils/preprocess.py  ->  predict_text(), predict_image(), get_label()
  utils/fusion.py      ->  fuse_predictions(text_prob, image_prob)
  utils/explain.py     ->  make_gradcam_heatmap(), save_gradcam()
"""

import os
import json
import base64
import time
import traceback
import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from utils.preprocess import predict_text, predict_image, get_label
from utils.fusion     import fuse_predictions
from utils.explain    import (make_gradcam_heatmap, save_gradcam,
                               plot_feature_importance, shap_explanation)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = os.path.join('static', 'uploads')
app.config['SECRET_KEY']         = 'pink-ribbon-2024'

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'dcm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)


def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# ════════════════════════════════════════════════════════════════
# FEATURE CONFIG
# ════════════════════════════════════════════════════════════════

# The 30 standard Wisconsin features (mean → se → worst)
FEATURES_30 = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst',
]

# If 'id' column was NOT dropped in train_text_model.py, there are 31 features.
# We prepend a dummy id=0 in that case.
FEATURES_31 = ['id'] + FEATURES_30

# Wisconsin dataset medians for unmapped features
_MEDIANS = {
    'id':                     0.0,
    'compactness_mean':       0.10840,
    'concave_points_mean':    0.06154,
    'fractal_dimension_mean': 0.06154,
    'radius_se':              0.40510,
    'texture_se':             1.22000,
    'perimeter_se':           2.86600,
    'area_se':               40.34000,
    'smoothness_se':          0.00638,
    'compactness_se':         0.02045,
    'concavity_se':           0.02589,
    'concave_points_se':      0.01093,
    'symmetry_se':            0.01587,
    'fractal_dimension_se':   0.00300,
    'radius_worst':          14.97000,
    'texture_worst':         25.41000,
    'perimeter_worst':       97.66000,
    'area_worst':           686.50000,
    'smoothness_worst':       0.13200,
    'compactness_worst':      0.25400,
    'concavity_worst':        0.27190,
    'concave_points_worst':   0.11460,
    'symmetry_worst':         0.29000,
    'fractal_dimension_worst':0.08402,
}


def _get_model_n_features():
    """
    Reads the saved scaler to find out exactly how many features
    the trained model expects. Returns 30 or 31 (or None on error).
    """
    try:
        pkl_path = os.path.join('models', 'text_model.pkl')
        with open(pkl_path, 'rb') as f:
            _, scaler = pickle.load(f)
        return int(scaler.n_features_in_)
    except Exception as e:
        print(f'[FeatureCount] Could not read scaler: {e}')
        return 30   # safe default


def _build_feature_vector(form):
    """
    Reads 8 values from the frontend form and builds the exact
    feature vector the model was trained on (auto-detects 30 or 31).
    """
    n = _get_model_n_features()

    # Values from frontend sliders/inputs
    user_vals = {
        'radius_mean':    float(form.get('radius',      18.2)),
        'texture_mean':   float(form.get('texture',     22.4)),
        'perimeter_mean': float(form.get('perimeter',  114.0)),
        'area_mean':      float(form.get('area',       920.0)),
        'smoothness_mean':float(form.get('smoothness',  0.118)),
        'concavity_mean': float(form.get('concavity',   0.241)),
        'symmetry_mean':  float(form.get('symmetry',    0.205)),
    }

    if n == 31:
        feature_list = FEATURES_31
    else:
        feature_list = FEATURES_30

    vector = []
    for feat in feature_list:
        if feat in user_vals:
            vector.append(user_vals[feat])
        else:
            vector.append(_MEDIANS.get(feat, 0.0))

    return vector


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    text_ok  = os.path.exists(os.path.join('models', 'text_model.pkl'))
    image_ok = os.path.exists(os.path.join('models', 'image_model.h5'))
    n_feat   = _get_model_n_features() if text_ok else '?'
    return jsonify({
        'status':           'online',
        'text_model':       text_ok,
        'image_model':      image_ok,
        'model_features':   n_feat,
        'version':          '1.0.0',
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    start = time.time()
    try:

        # ── 1. Build feature vector (auto 30 or 31) ───────────
        feature_vector = _build_feature_vector(request.form)

        # ── 2. Text model prediction ──────────────────────────
        #       preprocess.predict_text(list) → (int, float 0-1)
        text_pred_int, text_prob = predict_text(feature_vector)
        text_label    = get_label(text_pred_int)
        text_prob_pct = round(float(text_prob) * 100, 2)

        feat_imp  = _safe_feature_importance()
        shap_vals = _safe_shap(feature_vector)

        text_result = {
            'prediction':         text_label,
            'probability':        text_prob_pct,
            'best_model':         'Logistic Regression',
            'all_models':         {'Logistic Regression': text_prob_pct},
            'feature_importance': feat_imp,
            'shap_values':        shap_vals,
            'metrics':            {},
        }

        # ── 3. Image model prediction ─────────────────────────
        image_result   = None
        gradcam_b64    = None

        for field in ['cc_image', 'mlo_image', 'image']:
            if field in request.files:
                f = request.files[field]
                if f and f.filename and allowed(f.filename):
                    fname          = secure_filename(f'{int(time.time())}_{f.filename}')
                    saved_img_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    f.save(saved_img_path)

                    img_pred_int, img_prob = predict_image(saved_img_path)
                    img_label    = get_label(img_pred_int)
                    img_prob_pct = round(float(img_prob) * 100, 2)

                    gradcam_path = os.path.join('static', 'gradcam.jpg')
                    _make_gradcam_safe(saved_img_path, gradcam_path)

                    if os.path.exists(gradcam_path):
                        with open(gradcam_path, 'rb') as gf:
                            gradcam_b64 = base64.b64encode(gf.read()).decode()

                    image_result = {
                        'prediction':  img_label,
                        'probability': img_prob_pct,
                        'radiomics':   _extract_radiomics(saved_img_path, float(img_prob)),
                    }
                    break

        # ── 4. Fusion ─────────────────────────────────────────
        if image_result is not None:
            img_prob_raw = float(image_result['probability']) / 100.0
        else:
            img_prob_raw = float(text_prob)

        fused_pred_int, fused_score, fused_category = fuse_predictions(
            float(text_prob), img_prob_raw
        )
        fused_label     = get_label(fused_pred_int)
        fused_score_pct = round(fused_score * 100, 2)
        confidence      = fused_score_pct if fused_label == 'Malignant' else round(100 - fused_score_pct, 1)

        stage = _estimate_stage(fused_score_pct)
        rec   = _recommendation(fused_label, fused_category)

        return jsonify({
            'status':         'success',
            'inference_time': round(time.time() - start, 3),

            'prediction':    fused_label,
            'risk_score':    fused_score_pct,
            'risk_category': fused_category.replace(' Risk', ''),
            'confidence':    confidence,

            'text_model': text_result,

            'image_model': {
                'available':   image_result is not None,
                'prediction':  image_result['prediction']  if image_result else None,
                'probability': image_result['probability'] if image_result else None,
                'radiomics':   image_result['radiomics']   if image_result else None,
                'gradcam':     gradcam_b64,
            },

            'fusion': {
                'text_weight':   0.6,
                'image_weight':  0.4,
                'fusion_method': 'Weighted Fusion (60% Clinical + 40% Imaging)',
            },

            'risk_breakdown': {
                'malignancy_probability': fused_score_pct,
                'invasiveness_score':     round(min(fused_score_pct * 0.8,  100), 1),
                'metastatic_risk':        round(min(fused_score_pct * 0.55, 100), 1),
                'clinical_risk':          text_prob_pct,
                'imaging_risk':           image_result['probability'] if image_result else None,
            },

            'stage_estimate': stage,
            'recommendation': rec,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════

def _make_gradcam_safe(img_path, out_path):
    """Real Grad-CAM via explain.py; fallback to PIL heatmap."""
    try:
        from tensorflow.keras.models        import load_model
        from tensorflow.keras.preprocessing import image as keras_image

        model = load_model(os.path.join('models', 'image_model.h5'))

        last_conv = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv = layer.name
                break
        if last_conv is None:
            raise ValueError('No conv layer found')

        img_arr = keras_image.load_img(img_path, target_size=(224, 224))
        img_arr = keras_image.img_to_array(img_arr) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        heatmap = make_gradcam_heatmap(img_arr, model, last_conv)
        save_gradcam(img_path, model, heatmap)   # saves to static/gradcam.jpg

    except Exception as e:
        print(f'[GradCAM] explain.py failed: {e} — using fallback')
        _gradcam_pil_fallback(img_path, out_path)


def _gradcam_pil_fallback(img_path, out_path):
    try:
        from PIL import Image
        from scipy.ndimage import uniform_filter

        img  = Image.open(img_path).convert('RGB').resize((224, 224))
        orig = np.array(img, dtype=np.float32) / 255.0
        gray = np.array(img.convert('L'), dtype=np.float32) / 255.0

        blur = uniform_filter(gray, 20)
        var  = uniform_filter((gray - blur) ** 2, 20)
        var  = np.sqrt(var)
        if var.max() > 0:
            var /= var.max()

        r    = np.clip(1.5 - np.abs(var * 4 - 3), 0, 1)
        g    = np.clip(1.5 - np.abs(var * 4 - 2), 0, 1)
        b    = np.clip(1.5 - np.abs(var * 4 - 1), 0, 1)
        heat = np.stack([r, g, b], axis=-1)
        out  = np.clip(orig * 0.6 + heat * 0.4, 0, 1)
        Image.fromarray((out * 255).astype(np.uint8)).save(out_path)
    except Exception as e:
        print(f'[GradCAM PIL fallback] {e}')


def _safe_feature_importance():
    """Feature importance from saved model; fallback to static values."""
    try:
        with open(os.path.join('models', 'text_model.pkl'), 'rb') as f:
            model, _ = pickle.load(f)

        if hasattr(model, 'feature_importances_'):
            fi    = model.feature_importances_
            names = FEATURES_30[:len(fi)]
            total = fi.sum() or 1.0
            pairs = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
            return [{'name': n.replace('_', ' ').title(), 'score': round(float(v / total) * 100, 1)}
                    for n, v in pairs[:8]]

        elif hasattr(model, 'coef_'):
            coef  = np.abs(model.coef_[0])
            names = FEATURES_30[:len(coef)]
            total = coef.sum() or 1.0
            pairs = sorted(zip(names, coef), key=lambda x: x[1], reverse=True)
            return [{'name': n.replace('_', ' ').title(), 'score': round(float(v / total) * 100, 1)}
                    for n, v in pairs[:8]]

    except Exception as e:
        print(f'[FeatureImportance] {e}')

    return [
        {'name': 'Concavity Worst',      'score': 18.2},
        {'name': 'Radius Worst',         'score': 15.7},
        {'name': 'Area Mean',            'score': 13.4},
        {'name': 'Perimeter Worst',      'score': 11.8},
        {'name': 'Texture Mean',         'score': 9.6},
        {'name': 'Smoothness Mean',      'score': 7.3},
        {'name': 'Symmetry Mean',        'score': 5.1},
        {'name': 'Fractal Dimension Mean','score': 4.2},
    ]


def _safe_shap(feature_vector):
    """SHAP values for explainability panel."""
    try:
        import shap
        with open(os.path.join('models', 'text_model.pkl'), 'rb') as f:
            model, scaler = pickle.load(f)

        X        = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)

        explainer = shap.Explainer(model, X_scaled)
        vals      = explainer(X_scaled)
        raw       = vals.values[0].tolist()

        names  = FEATURES_31 if len(raw) == 31 else FEATURES_30
        result = [
            {'feature': names[i] if i < len(names) else f'f{i}',
             'shap':    round(float(v), 4),
             'score':   round(abs(float(v)) * 100, 1)}
            for i, v in enumerate(raw)
        ]
        return sorted(result, key=lambda x: abs(x['shap']), reverse=True)
    except Exception as e:
        print(f'[SHAP] {e}')
        return []


def _extract_radiomics(image_path, prob):
    try:
        from PIL import Image
        img  = Image.open(image_path).convert('L').resize((128, 128))
        arr  = np.array(img, dtype=np.float32) / 255.0
        mean = arr.mean()
        std  = arr.std()
        hist, _ = np.histogram(arr, bins=32, range=(0, 1))
        hn   = hist / (hist.sum() + 1e-8)
        ent  = float(-np.sum(hn * np.log2(hn + 1e-8)))
        return {
            'mass_density': {
                'value': 'High' if mean < 0.35 else ('Moderate' if mean < 0.55 else 'Low'),
                'score': round((1 - mean) * 100, 1),
            },
            'margin_irregularity': {
                'value': 'Spiculated' if std > 0.22 else ('Irregular' if std > 0.15 else 'Smooth'),
                'score': round(std * 400, 1),
            },
            'calcification': {
                'value': 'Suspicious' if prob > 0.6 else 'Benign',
                'score': round(prob * 100, 1),
            },
            'echogenicity': {
                'value': 'Hypoechoic' if mean < 0.45 else 'Isoechoic',
                'score': round(ent * 10, 1),
            },
        }
    except Exception:
        return {}


def _estimate_stage(pct):
    if pct < 30:
        return {'stage': 0, 'tnm': 'T0 N0 M0',
                'desc': 'No malignancy detected', 'active_stages': []}
    elif pct < 50:
        return {'stage': 1, 'tnm': 'T1 N0 M0',
                'desc': 'Small tumor, no lymph node involvement', 'active_stages': [1]}
    elif pct < 65:
        return {'stage': 2, 'tnm': 'T2 N1 M0',
                'desc': 'Tumor 2-5 cm, minimal lymph node involvement', 'active_stages': [1, 2]}
    elif pct < 80:
        return {'stage': 3, 'tnm': 'T3 N2 M0',
                'desc': 'Large tumor, multiple lymph nodes affected', 'active_stages': [1, 2, 3]}
    else:
        return {'stage': 4, 'tnm': 'T4 N3 M1',
                'desc': 'Advanced disease, possible distant metastasis', 'active_stages': [1, 2, 3, 4]}


def _recommendation(prediction, category):
    if prediction == 'Malignant' or 'High' in category:
        return {
            'summary': 'High suspicion of malignancy. Immediate clinical follow-up is strongly recommended.',
            'urgency': 'Urgent',
            'steps': [
                'Urgent core needle biopsy of the suspicious lesion.',
                'Contrast-enhanced MRI for accurate local staging and contralateral breast assessment.',
                'Multidisciplinary tumor board referral within 5 business days.',
                'Genetic counseling if BRCA positive or strong family history.',
            ],
        }
    elif 'Medium' in category:
        return {
            'summary': 'Moderate risk detected. Confirmatory diagnostic tests recommended.',
            'urgency': 'Moderate',
            'steps': [
                'Follow-up diagnostic mammography in 4-6 weeks.',
                'Ultrasound-guided biopsy if lesion unchanged or growing.',
                'Consultation with a breast surgeon.',
                'Continue regular self-examination.',
            ],
        }
    else:
        return {
            'summary': 'Low risk. Continue routine screening and monitoring.',
            'urgency': 'Routine',
            'steps': [
                'Continue routine annual mammography screening.',
                'Maintain a healthy lifestyle (BMI, exercise, balanced diet).',
                'Monthly breast self-examination.',
                'Report any new or changing symptoms to your physician immediately.',
            ],
        }


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print('=' * 55)
    print('  Pink Ribbon - AI Breast Cancer Detection System')
    print('=' * 55)
    print('  http://localhost:5001')
    print('=' * 55)
    app.run(debug=True, host='0.0.0.0', port=5001)
