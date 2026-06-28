"""
LaTeX Vocabulary for Handwritten Formula Recognition
直接使用 BTTR 原仓库的 dictionary.txt (CROHME 数据集词表)

Token order:
  0: <pad> (PAD_IDX)
  1: <sos> (SOS_IDX)
  2: <eos> (EOS_IDX)
  3-113: CROHME LaTeX tokens (from dictionary.txt)
"""
import os
from typing import List


class LatexVocabulary:
    """CROHME LaTeX token vocabulary — matches BTTR pretrained checkpoint."""

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = None):
        # Build from dictionary.txt (original BTTR vocab file)
        self.word2idx = {
            "<pad>": self.PAD_IDX,
            "<sos>": self.SOS_IDX,
            "<eos>": self.EOS_IDX,
        }

        if dict_path is None:
            # Try to find dictionary.txt relative to this file
            dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")
            if not os.path.exists(dict_path):
                # Fallback: look in BTTR_original clone
                alt_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../../BTTR_original/bttr/datamodule/dictionary.txt"
                )
                if os.path.exists(alt_path):
                    dict_path = alt_path

        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    token = line.strip()
                    if token and token not in self.word2idx:
                        self.word2idx[token] = len(self.word2idx)
        else:
            # Hardcoded fallback: the exact 113-token CROHME dictionary
            self._build_default()

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.n_tokens = len(self.word2idx)

    def _build_default(self):
        """Fallback: hardcoded CROHME dictionary (113 tokens total incl. special)."""
        tokens = [
            "!", "'", "(", ")", "+", ",", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "<", "=", ">",
            "A", "B", "C", "E", "F", "G", "H", "I", "L", "M",
            "N", "P", "R", "S", "T", "V", "X", "Y",
            "[", "\\Delta", "\\alpha", "\\beta", "\\cdot", "\\cdots",
            "\\cos", "\\div", "\\exists", "\\forall", "\\frac", "\\gamma",
            "\\geq", "\\in", "\\infty", "\\int", "\\lambda", "\\ldots",
            "\\leq", "\\lim", "\\limits", "\\log", "\\mu", "\\neq",
            "\\phi", "\\pi", "\\pm", "\\prime", "\\rightarrow", "\\sigma",
            "\\sin", "\\sqrt", "\\sum", "\\tan", "\\theta", "\\times",
            "\\{", "\\}", "]", "^", "_",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z",
            "{", "|", "}",
        ]
        for t in tokens:
            if t not in self.word2idx:
                self.word2idx[t] = len(self.word2idx)

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx.get(w, self.word2idx["<pad>"]) for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word.get(i, "<pad>") for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        """Convert token indices to LaTeX string (CROHME format: space-separated)."""
        words = []
        for i in id_list:
            if i == self.EOS_IDX:
                break
            if i in [self.PAD_IDX, self.SOS_IDX]:
                continue
            words.append(self.idx2word.get(i, ""))
        return " ".join(words)

    def tokenize(self, latex: str) -> List[int]:
        """Tokenize a LaTeX string to token indices."""
        return self.words2indices(latex.split())

    def state_dict(self) -> dict:
        return {"word2idx": self.word2idx}

    def load_state_dict(self, state: dict) -> None:
        self.word2idx = {str(k): int(v) for k, v in state.get("word2idx", {}).items()}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.n_tokens = len(self.word2idx)

    def __len__(self):
        return self.n_tokens


# Global instance
latex_vocab = LatexVocabulary()
