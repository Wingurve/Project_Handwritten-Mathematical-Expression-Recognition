"""
Image-to-Formula Inference Pipeline

完整的手写公式识别推理管线:
  1. 图像预处理 (支持 png/jpg/bmp/gif/webp 等格式)
  2. BTTR 模型推理 (图像 → LaTeX token序列)
  3. LaTeX 解码 (token序列 → LaTeX 字符串)
  4. (可选) DeepSeek LLM 后处理纠错
"""
import os
import io
import torch
import numpy as np
from typing import Optional
from PIL import Image

from llm_corrector import LitBTTR
from llm_corrector.datamodule.latex_vocab import latex_vocab
from llm_corrector.checkpoint_loader import load_bttr_checkpoint


class FormulaRecognizer:
    """
    Handwritten formula recognizer.
    Image → BTTR Model → LaTeX → (optional) LLM correction → Final LaTeX
    """

    def __init__(
        self,
        checkpoint_path: str = "pretrained-2014.ckpt",
        device: str = None,
        use_llm_correction: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_llm_correction = use_llm_correction
        self.last_preprocess_info = {}

        print(f"Loading BTTR model from {checkpoint_path}...")
        self.model = LitBTTR(
            d_model=256,
            growth_rate=24,
            num_layers=16,
            nhead=8,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.3,
            beam_size=10,
            max_len=200,
            alpha=1.0,
        )
        load_bttr_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}, vocab size: {latex_vocab.n_tokens}")

        self.llm_corrector = None
        if use_llm_correction:
            self._init_llm_corrector()

    def _init_llm_corrector(self):
        try:
            from deepseek_client import DeepSeekEmbedder
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if api_key:
                self.llm_corrector = DeepSeekEmbedder(api_key=api_key)
                print("DeepSeek LLM corrector initialized")
        except Exception as e:
            print(f"LLM corrector init failed: {e}")

    def preprocess_image(
        self,
        image: Image.Image,
        max_pixels: int = 320000,  # BTTR MAX_SIZE = 32e4
        padding: int = 24,
    ) -> tuple:
        """
        Preprocess image to CROHME-like format for BTTR model.

        Original BTTR uses torchvision.transforms.ToTensor() on CROHME BMP images.
        CROHME images are dark background with light strokes, so this method:
        - converts to grayscale
        - inverts light-background photos
        - crops large blank borders
        - preserves grayscale instead of hard-binarizing

        Returns
        -------
        img : torch.Tensor [1, 1, H, W] - grayscale image, dark bg, values 0-1
        img_mask : torch.Tensor [1, H, W] - Bool mask
        """
        original_size = image.size

        # Step 1: Convert to grayscale. Composite transparent images onto white.
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image.convert("RGBA"))
            image = background.convert("L")
        if image.mode != 'L':
            image = image.convert('L')

        img_array = np.array(image, dtype=np.float32)

        # Step 2: Estimate background from image border and invert white-paper photos.
        border = np.concatenate([
            img_array[0, :],
            img_array[-1, :],
            img_array[:, 0],
            img_array[:, -1],
        ])
        border_median = float(np.median(border))
        inverted = border_median > 127
        if inverted:
            img_array = 255.0 - img_array

        # Step 3: Crop large blank borders after polarity normalization.
        # Keep grayscale values; use a stricter stroke map so notebook grid lines
        # do not dominate the crop box.
        norm = img_array / 255.0
        positive = norm[norm > 0.02]
        crop_threshold = None
        min_crop_threshold = 0.55 if inverted else 0.35
        if positive.size:
            crop_threshold = max(
                min_crop_threshold,
                float(norm.mean() + norm.std() * 2.0),
                float(np.percentile(positive, 99.4)),
            )
            foreground = norm >= crop_threshold
        else:
            foreground = norm > 0.05

        if foreground.sum() < 8 and positive.size:
            crop_threshold = max(
                0.35 if inverted else 0.18,
                float(norm.mean() + norm.std() * 1.25),
                float(np.percentile(positive, 96)),
            )
            foreground = norm >= crop_threshold

        if foreground.any():
            foreground = self._dilate_mask(foreground, iterations=2)
            ys, xs = np.where(foreground)
            y0 = max(int(ys.min()) - padding, 0)
            y1 = min(int(ys.max()) + padding + 1, norm.shape[0])
            x0 = max(int(xs.min()) - padding, 0)
            x1 = min(int(xs.max()) + padding + 1, norm.shape[1])
            norm = norm[y0:y1, x0:x1]

        # Step 4: Scale down if too large, preserving grayscale.
        h, w = norm.shape
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            norm_img = Image.fromarray(np.uint8(np.clip(norm * 255.0, 0, 255)))
            norm_img = norm_img.resize((new_w, new_h), Image.LANCZOS)
            norm = np.array(norm_img, dtype=np.float32) / 255.0

        # Convert to tensor [1, 1, H, W]
        img = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        # Mask: all False = all positions valid
        img_mask = torch.zeros(img.size(0), img.size(2), img.size(3),
                               dtype=torch.bool, device=self.device)

        self.last_preprocess_info = {
            "original_size": original_size,
            "processed_size": (int(img.size(3)), int(img.size(2))),
            "inverted": inverted,
            "crop_threshold": crop_threshold,
            "mean": float(img.mean().detach().cpu()),
        }

        return img, img_mask

    @staticmethod
    def _dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Small numpy-only dilation to make the crop box less brittle."""
        out = mask.astype(bool)
        for _ in range(iterations):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            out = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
                | padded[:-2, :-2]
                | padded[:-2, 2:]
                | padded[2:, :-2]
                | padded[2:, 2:]
            )
        return out

    def recognize(
        self,
        image: Image.Image,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> dict:
        """Recognize formula from PIL Image."""
        img, img_mask = self.preprocess_image(image)

        with torch.no_grad():
            hyps = self.model.bttr.beam_search(img, img_mask, beam_size, max_len)
            best_hyp = max(hyps, key=lambda h: h.score / (max(1, len(h)) ** alpha))
            raw_latex = latex_vocab.indices2label(best_hyp.seq)
            normalized_score = best_hyp.score / max(1, len(best_hyp))

        # Convert CROHME space-separated format to standard LaTeX
        standard_latex = self._to_standard_latex(raw_latex)

        result = {
            "latex": raw_latex,
            "latex_standard": standard_latex,
            "corrected_latex": None,
            "llm_enabled": self.llm_corrector is not None,
            "raw_tokens": best_hyp.seq,
            "confidence": float(normalized_score),
            "preprocess": self.last_preprocess_info,
        }

        if self.llm_corrector is not None:
            try:
                instruction = (
                    "请修正以下 LaTeX 数学公式中的明显识别错误，只输出可直接渲染的 LaTeX 代码。"
                    "不要解释，不要使用 Markdown，不要添加美元符号。"
                    "如果无法确定，请尽量保持原公式结构。"
                )
                corrected = self.llm_corrector.correct_with_deepseek(
                    standard_latex or raw_latex,
                    instruction=instruction,
                )
                result["corrected_latex"] = corrected
            except Exception as e:
                result["corrected_error"] = str(e)

        return result

    @staticmethod
    def _to_standard_latex(crohme_latex: str) -> str:
        """Convert CROHME space-separated LaTeX to standard format.
        Example: '\sqrt { 4 8 }' → '\\sqrt{48}'
        """
        # Remove spaces between single-character non-letter tokens
        result = []
        tokens = crohme_latex.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith('\\'):
                # LaTeX command - keep as-is, add space if next is also a command
                result.append(tok)
                i += 1
            else:
                # Single-char tokens: group consecutive ones
                group = []
                while i < len(tokens) and not tokens[i].startswith('\\'):
                    group.append(tokens[i])
                    i += 1
                result.append(''.join(group))
        return ''.join(result)

    def recognize_from_bytes(self, image_bytes: bytes, **kwargs) -> dict:
        return self.recognize(Image.open(io.BytesIO(image_bytes)), **kwargs)

    def recognize_from_path(self, path: str, **kwargs) -> dict:
        return self.recognize(Image.open(path), **kwargs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        recognizer = FormulaRecognizer()
        result = recognizer.recognize_from_path(sys.argv[1])
        print(f"LaTeX: {result['latex']}")
        if result['corrected_latex']:
            print(f"Corrected: {result['corrected_latex']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print("Usage: python image_inference.py <image_path>")
