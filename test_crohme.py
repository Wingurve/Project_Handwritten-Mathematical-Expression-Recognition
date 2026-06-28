"""
CROHME 2014/2016/2019 Test Evaluation
"""
import sys
import os
import torch
import editdistance
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image
from torchvision.transforms import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BTTR_original'))
from bttr.datamodule import vocab as orig_vocab

from llm_corrector import LitBTTR
from llm_corrector.datamodule.latex_vocab import latex_vocab
from llm_corrector.checkpoint_loader import load_bttr_checkpoint


def load_test_data(data_zip_path, year="2014"):
    """Load CROHME test data from zip (matching original extract_data)."""
    data = []
    with ZipFile(data_zip_path) as archive:
        with archive.open(f"{year}/caption.txt", "r") as f:
            captions = f.readlines()
        for line in captions:
            tmp = line.decode().strip().split()
            img_name = tmp[0]
            formula = tmp[1:]  # space-separated CROHME tokens
            img_path = f"{year}/{img_name}.bmp"
            if img_path in archive.namelist():
                with archive.open(img_path, "r") as f:
                    img = Image.open(f).copy()
                data.append((img_name, img, formula))
            else:
                print(f"  WARNING: {img_path} not found in zip")
    return data


def evaluate(year="2014", max_samples=None, beam_size=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading BTTR model...")
    model = LitBTTR(
        d_model=256, growth_rate=24, num_layers=16,
        nhead=8, num_decoder_layers=3, dim_feedforward=1024,
        dropout=0.3, beam_size=beam_size, max_len=200, alpha=1.0,
    )
    load_bttr_checkpoint("pretrained-2014.ckpt", model, map_location=device)
    model.to(device)
    model.eval()

    # Load data
    data_zip = os.path.join(os.path.dirname(__file__), '..', 'BTTR_original', 'data.zip')
    print(f"Loading CROHME {year} test data...")
    test_data = load_test_data(data_zip, year)
    print(f"Test samples: {len(test_data)}")

    if max_samples:
        test_data = test_data[:max_samples]
        print(f"Using first {max_samples} samples")

    total = 0
    correct = 0
    results = []

    for img_name, img, formula_tokens in tqdm(test_data, desc=f"CROHME {year}"):
        # Preprocess: PIL → ToTensor → [1,1,H,W]
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        img_mask = torch.zeros(1, img_tensor.size(2), img_tensor.size(3),
                               dtype=torch.bool, device=device)

        # Inference
        with torch.no_grad():
            hyps = model.bttr.beam_search(img_tensor, img_mask, beam_size, 200)
            best_hyp = max(hyps, key=lambda h: h.score / (max(1, len(h)) ** 1.0))
            pred_tokens = best_hyp.seq

        # Ground truth
        gt_tokens = orig_vocab.words2indices(formula_tokens)

        # ExpRate: exact match (edit distance == 0)
        dist = editdistance.eval(pred_tokens, gt_tokens)
        is_correct = (dist == 0)
        if is_correct:
            correct += 1
        total += 1

        if len(results) < 20:
            results.append({
                'name': img_name,
                'pred': latex_vocab.indices2label(pred_tokens),
                'gt': orig_vocab.indices2label(gt_tokens),
                'correct': is_correct,
                'dist': dist,
            })

    exp_rate = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"CROHME {year} Test Results")
    print(f"{'='*60}")
    print(f"Total: {total}  |  Correct: {correct}  |  ExpRate: {exp_rate:.2f}%")
    print(f"{'='*60}")

    print(f"\nFirst 20 predictions:")
    for r in results:
        s = "✓" if r['correct'] else f"✗(d={r['dist']})"
        print(f"  [{s}] {r['name']}")
        print(f"    Pred: {r['pred']}")
        print(f"    GT:   {r['gt']}")
        print()

    return exp_rate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2014")
    parser.add_argument("--max", type=int, default=None, help="Max samples (for quick test)")
    parser.add_argument("--beam", type=int, default=10)
    args = parser.parse_args()

    evaluate(year=args.year, max_samples=args.max, beam_size=args.beam)
