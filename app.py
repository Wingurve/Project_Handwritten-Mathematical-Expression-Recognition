"""
Flask Web Application for Handwritten Formula Recognition
Image upload → BTTR model → LaTeX output
"""
import os
import io
import base64
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Initialize Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global recognizer (loaded lazily)
_recognizer = None


def get_recognizer():
    """Lazy-load the formula recognizer."""
    global _recognizer
    if _recognizer is None:
        from image_inference import FormulaRecognizer
        checkpoint = os.path.join(os.path.dirname(__file__), "pretrained-2014.ckpt")
        _recognizer = FormulaRecognizer(
            checkpoint_path=checkpoint,
            device="cpu",
            use_llm_correction=True,
        )
    return _recognizer


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff', 'tif', 'ico'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Recognize formula from uploaded image.

    Accepts:
    - multipart/form-data with 'image' field
    - JSON with 'image' as base64 string

    Returns JSON:
    {
        "success": true,
        "latex": "\\frac{a}{b}",
        "corrected_latex": null,
        "confidence": -2.5,
        "image_base64": "data:image/png;base64,..."
    }
    """
    try:
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": f"Unsupported file type. Supported: PNG, JPG, BMP, GIF, WebP"}), 400

            image_bytes = file.read()
            if len(image_bytes) == 0:
                return jsonify({"success": False, "error": "Empty file"}), 400

        # Handle base64 image in JSON
        elif request.is_json:
            data = request.get_json()
            image_data = data.get('image', '')
            if image_data.startswith('data:'):
                # Remove data URL prefix
                image_data = image_data.split(',', 1)[1]
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                return jsonify({"success": False, "error": "Invalid base64 image"}), 400
        else:
            return jsonify({"success": False, "error": "No image provided. Upload a file or send base64 JSON."}), 400

        # Convert to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # Detect mime type
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes))
        mime = f"image/{img.format.lower()}" if img.format else "image/png"
        image_data_uri = f"data:{mime};base64,{image_base64}"

        # Run recognition
        recognizer = get_recognizer()
        result = recognizer.recognize_from_bytes(image_bytes)

        return jsonify({
            "success": True,
            "latex": result.get("latex_standard", result.get("latex", "")),
            "latex_raw": result.get("latex", ""),
            "corrected_latex": result.get("corrected_latex"),
            "corrected_error": result.get("corrected_error"),
            "llm_enabled": result.get("llm_enabled", False),
            "confidence": round(result.get("confidence", 0), 2),
            "preprocess": result.get("preprocess", {}),
            "image_base64": image_data_uri,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": _recognizer is not None,
        "llm_enabled": bool(_recognizer and _recognizer.llm_corrector is not None),
    })


if __name__ == '__main__':
    # Warm up: load model on startup
    print("Warming up: loading model...")
    try:
        get_recognizer()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Model preload failed: {e}")
        print("Model will be loaded on first request.")

    app.run(host='0.0.0.0', port=5000, debug=True)
