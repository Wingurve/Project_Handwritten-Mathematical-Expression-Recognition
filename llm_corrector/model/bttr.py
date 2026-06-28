"""
BTTR Model — exact match of original BTTR (Green-Wood/BTTR).

Architecture:
  Image → DenseNetEncoder → feature [b, t, d] → Decoder → LaTeX tokens
                                                     ↓
                                           bidirectional beam search
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor

from .densenet_encoder import DenseNetEncoder
from .pos_enc import WordPosEnc
from ..datamodule.latex_vocab import latex_vocab


# ========================
# Hypothesis
# ========================
class Hypothesis:
    """Beam search hypothesis"""

    def __init__(self, seq_tensor: torch.Tensor, score: float, direction: str):
        self.seq_tensor = seq_tensor
        self.score = score
        self.direction = direction

    def __len__(self):
        return len(self.seq)

    @property
    def seq(self) -> List[int]:
        return self.seq_tensor.tolist()


# ========================
# Helper: build target/output for cross-rate scoring
# ========================
def to_tgt_output(indices, direction, device):
    """Prepare target and output for cross-rate scoring."""
    assert direction in {"l2r", "r2l"}
    SOS = latex_vocab.SOS_IDX
    EOS = latex_vocab.EOS_IDX
    PAD = latex_vocab.PAD_IDX

    if direction == "l2r":
        max_len = max(len(idx) for idx in indices) + 2
        tgt = torch.full((len(indices), max_len), fill_value=PAD, dtype=torch.long, device=device)
        out = torch.full_like(tgt, fill_value=PAD)
        for i, seq in enumerate(indices):
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            tgt[i, 0] = SOS
            tgt[i, 1:len(seq)+1] = seq_t
            out[i, :len(seq)] = seq_t
            out[i, len(seq)] = EOS
    else:
        max_len = max(len(idx) for idx in indices) + 2
        tgt = torch.full((len(indices), max_len), fill_value=PAD, dtype=torch.long, device=device)
        out = torch.full_like(tgt, fill_value=PAD)
        for i, seq in enumerate(indices):
            rev = list(reversed(seq))
            rev_t = torch.tensor(rev, dtype=torch.long, device=device)
            tgt[i, 0] = EOS
            tgt[i, 1:len(rev)+1] = rev_t
            out[i, :len(rev)] = rev_t
            out[i, len(rev)] = SOS

    return tgt, out


# ========================
# Decoder (matching original BTTR Decoder)
# ========================
class Decoder(nn.Module):
    """Transformer decoder matching original BTTR Decoder.

    Uses batch_first=False internally (original PyTorch TransformerDecoder convention).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        vocab_size = latex_vocab.n_tokens

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = WordPosEnc(d_model=d_model, max_len=500)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length, device):
        mask = torch.full((length, length), fill_value=1, dtype=torch.bool, device=device)
        mask.triu_(1)
        return mask

    def forward(self, src, src_mask, tgt):
        """
        Parameters
        ----------
        src : [b, t, d]
        src_mask : [b, t] - bool, True=padded
        tgt : [b, l] - token indices

        Returns
        -------
        [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l, tgt.device)
        tgt_pad_mask = (tgt == latex_vocab.PAD_IDX)

        tgt_emb = self.word_embed(tgt)        # [b, l, d]
        tgt_emb = self.pos_enc(tgt_emb)       # [b, l, d]

        # TransformerDecoder expects (T, B, D) format
        src_t = rearrange(src, "b t d -> t b d")
        tgt_t = rearrange(tgt_emb, "b l d -> l b d")

        out = self.model(
            tgt=tgt_t,
            memory=src_t,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        return self.proj(out)

    def _beam_search(self, src, mask, direction, beam_size, max_len):
        """Beam search for one direction (l2r or r2l)."""
        assert direction in {"l2r", "r2l"}
        assert src.size(0) == 1 and mask.size(0) == 1

        if direction == "l2r":
            start_w = latex_vocab.SOS_IDX
            stop_w = latex_vocab.EOS_IDX
        else:
            start_w = latex_vocab.EOS_IDX
            stop_w = latex_vocab.SOS_IDX

        hypotheses = torch.full(
            (1, max_len + 1), fill_value=latex_vocab.PAD_IDX,
            dtype=torch.long, device=src.device
        )
        hypotheses[:, 0] = start_w
        hyp_scores = torch.zeros(1, dtype=torch.float, device=src.device)
        completed: List[Hypothesis] = []

        t = 0
        while len(completed) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size

            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)

            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]
            log_p_t = F.log_softmax(decode_outputs, dim=-1)

            live_hyp_num = beam_size - len(completed)
            V = latex_vocab.n_tokens
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=V)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            top_scores, top_pos = torch.topk(continuous_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_pos // V
            hyp_word_ids = top_pos % V

            t += 1
            new_hyps = []
            new_scores = []

            for prev_id, word_id, score in zip(prev_hyp_ids, hyp_word_ids, top_scores):
                score = score.detach().item()
                hypotheses[prev_id, t] = word_id

                if word_id == stop_w:
                    completed.append(Hypothesis(
                        seq_tensor=hypotheses[prev_id, 1:t].detach().clone(),
                        score=score,
                        direction=direction,
                    ))
                else:
                    new_hyps.append(hypotheses[prev_id].detach().clone())
                    new_scores.append(score)

            if len(completed) >= beam_size:
                break

            if not new_hyps:
                break

            hypotheses = torch.stack(new_hyps, dim=0)
            hyp_scores = torch.tensor(new_scores, dtype=torch.float, device=src.device)

        if not completed:
            completed.append(Hypothesis(
                seq_tensor=hypotheses[0, 1:].detach().clone(),
                score=hyp_scores[0].detach().item(),
                direction=direction,
            ))

        return completed

    def _cross_rate_score(self, src, mask, hypotheses, direction):
        """Score L2R hypotheses with R2L model (and vice versa)."""
        if not hypotheses:
            return
        indices = [h.seq for h in hypotheses]
        tgt, output = to_tgt_output(indices, direction, src.device)

        b = tgt.size(0)
        exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
        exp_mask = repeat(mask.squeeze(0), "s -> b s", b=b)

        output_hat = self(exp_src, exp_mask, tgt)

        flat_hat = rearrange(output_hat, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        loss = F.cross_entropy(flat_hat, flat, ignore_index=latex_vocab.PAD_IDX, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, l in enumerate(loss):
            hypotheses[i].score += (-l).detach().item()

    def beam_search(self, src, mask, beam_size, max_len):
        """Bidirectional beam search."""
        l2r = self._beam_search(src, mask, "l2r", beam_size, max_len)
        self._cross_rate_score(src, mask, l2r, direction="r2l")

        r2l = self._beam_search(src, mask, "r2l", beam_size, max_len)
        self._cross_rate_score(src, mask, r2l, direction="l2r")

        return l2r + r2l


# ========================
# BTTR Full Model
# ========================
class BTTR(nn.Module):
    """BTTR: Bidirectionally Trained Transformer for HMER."""

    def __init__(
        self,
        d_model: int = 256,
        growth_rate: int = 24,
        num_layers: int = 16,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = DenseNetEncoder(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, img, img_mask, tgt):
        """
        Parameters
        ----------
        img : [b, 1, h', w']
        img_mask : [b, h', w'] - bool, True=padded
        tgt : [2b, l] - bidirectional targets

        Returns
        -------
        [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)      # [b, t, d], [b, t]
        feature = torch.cat((feature, feature), dim=0)    # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)             # [2b, t]
        return self.decoder(feature, mask, tgt)

    def beam_search(self, img, img_mask, beam_size, max_len):
        """
        Parameters
        ----------
        img : [1, 1, h', w']
        img_mask : [1, h', w'] - bool, True=padded
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [1, t, d], [1, t]
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
