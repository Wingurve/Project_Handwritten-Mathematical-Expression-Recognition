# Project Handwritten Mathematical Expression Recognition

FormulaReader is a local web application for handwritten mathematical expression recognition. It uses a pretrained BTTR model to convert formula images into LaTeX, then optionally calls DeepSeek to clean up obvious LaTeX recognition errors.

## Pipeline

```text
image upload
  -> image preprocessing
  -> BTTR handwritten formula recognizer
  -> raw LaTeX
  -> DeepSeek LaTeX post-processing
  -> final LaTeX + rendered preview
```

The LLM does not read the image directly. It only post-processes the LaTeX string produced by BTTR.

## Features

- Handwritten formula image recognition with BTTR.
- Local Flask web UI for upload, drag-and-drop, and clipboard paste.
- KaTeX preview for both BTTR output and LLM-corrected output.
- DeepSeek post-processing for more renderable LaTeX.
- Preprocessing for white-paper photos, grid paper, background inversion, cropping, and resizing.
- CPU-first setup, with no CUDA requirement.

## Project Structure

```text
.
├── app.py                         # Flask API and web server
├── image_inference.py             # Image preprocessing + BTTR + DeepSeek pipeline
├── deepseek_client.py             # DeepSeek LaTeX post-processing client
├── pretrained-2014.ckpt           # BTTR pretrained checkpoint
├── test_crohme.py                 # CROHME smoke/evaluation script
├── templates/index.html           # Web UI
├── static/css/style.css           # Web UI styles
└── llm_corrector/
    ├── checkpoint_loader.py       # Checkpoint loading/remapping
    ├── lit_corrector.py           # LitBTTR Lightning wrapper
    ├── datamodule/latex_vocab.py  # CROHME LaTeX vocabulary
    └── model/
        ├── bttr.py                # BTTR model and decoder
        ├── densenet_encoder.py    # DenseNet image encoder
        └── pos_enc.py             # Decoder word positional encoding
```

## Requirements

- Python 3.8+
- PyTorch
- Flask
- Pillow
- OpenAI Python SDK for DeepSeek's OpenAI-compatible API

Install dependencies:

```bash
pip install -r requirements.txt
```

## DeepSeek Setup

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

If `DEEPSEEK_API_KEY` is missing, the app still returns BTTR output, but the LLM correction panel will show that post-processing is unavailable.

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The current app runs on CPU by default. This is intentional for portability.

## API

### Health Check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "llm_enabled": true
}
```

### Recognize Formula

```http
POST /api/recognize
```

Supported input:

- `multipart/form-data` with an `image` file field
- JSON with a base64 `image` field

Example response:

```json
{
  "success": true,
  "latex": "x_{k}xx_{k}+y_{k}yx_{k}",
  "latex_raw": "x _ { k } x x _ { k } + y _ { k } y x _ { k }",
  "corrected_latex": "x_{k} x_{k} + y_{k} y_{k}",
  "llm_enabled": true,
  "confidence": -0.19,
  "preprocess": {
    "original_size": [464, 85],
    "processed_size": [457, 78],
    "inverted": false
  }
}
```

## Input Tips

BTTR was trained on CROHME-style formula images: single formulas, tight crops, clean background, and high contrast. Recognition quality drops when the uploaded image contains:

- a full notebook page instead of a cropped formula,
- grid lines stronger than handwriting,
- shadows, blur, or skew,
- multiple lines or non-math text,
- symbols outside the CROHME vocabulary.

For best results, crop the image close to the formula before uploading.

## Notes

- `pretrained-2014.ckpt` is expected in the project root.
- The project intentionally removes the previous text-error-correction training path. The only LLM functionality kept is LaTeX post-processing after BTTR image recognition.
- The pretrained model is based on the BTTR architecture by Green-Wood.
