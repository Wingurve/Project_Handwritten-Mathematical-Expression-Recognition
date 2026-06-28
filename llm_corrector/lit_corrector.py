"""PyTorch Lightning wrapper for BTTR handwritten formula recognition."""
import zipfile
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .datamodule import latex_vocab
from .model.bttr import BTTR


def to_bi_tgt_out(indices, device: torch.device):
    """Build bidirectional decoder input/output from CROHME token lists."""
    batch_size = len(indices)
    max_len = max(len(seq) for seq in indices)
    seq_len = max_len + 1

    tgt_l2r = torch.full(
        (batch_size, seq_len),
        fill_value=latex_vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out_l2r = torch.full_like(tgt_l2r, fill_value=latex_vocab.PAD_IDX)
    tgt_r2l = torch.full_like(tgt_l2r, fill_value=latex_vocab.PAD_IDX)
    out_r2l = torch.full_like(tgt_l2r, fill_value=latex_vocab.PAD_IDX)

    for i, seq in enumerate(indices):
        seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        rev_t = torch.flip(seq_t, dims=[0])
        length = len(seq)

        tgt_l2r[i, 0] = latex_vocab.SOS_IDX
        tgt_l2r[i, 1:length + 1] = seq_t
        out_l2r[i, :length] = seq_t
        out_l2r[i, length] = latex_vocab.EOS_IDX

        tgt_r2l[i, 0] = latex_vocab.EOS_IDX
        tgt_r2l[i, 1:length + 1] = rev_t
        out_r2l[i, :length] = rev_t
        out_r2l[i, length] = latex_vocab.SOS_IDX

    return torch.cat([tgt_l2r, tgt_r2l], dim=0), torch.cat([out_l2r, out_r2l], dim=0)


def ce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1, pred.size(-1))
    target = target.reshape(-1)
    return nn.CrossEntropyLoss(ignore_index=latex_vocab.PAD_IDX)(pred, target)


class CorrectRateRecorder:
    """Exact-match rate recorder for CROHME expression recognition."""

    def __init__(self):
        self.total = 0
        self.correct = 0

    def __call__(self, pred_seq: List[int], target_seq: List[int]):
        self.total += 1
        if pred_seq == target_seq:
            self.correct += 1

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        self.total = 0
        self.correct = 0


class LitBTTR(pl.LightningModule):
    """Lightning module for BTTR: image -> LaTeX handwritten formula recognition."""

    def __init__(
        self,
        d_model: int = 256,
        growth_rate: int = 24,
        num_layers: int = 16,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
        learning_rate: float = 1.0,
        patience: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bttr = BTTR(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.exprate_recorder = CorrectRateRecorder()
        self.test_outputs = []

    def forward(self, img, img_mask, tgt):
        return self.bttr(img, img_mask, tgt)

    def training_step(self, batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)
        loss = ce_loss(out_hat, out)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.imgs.size(0),
        )
        return loss

    def on_validation_epoch_start(self):
        self.exprate_recorder.reset()

    def validation_step(self, batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)
        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.imgs.size(0),
        )

        hyps = self.bttr.beam_search(
            batch.imgs,
            batch.mask,
            self.hparams.beam_size,
            self.hparams.max_len,
        )
        best_hyp = max(hyps, key=lambda h: h.score / (max(1, len(h)) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])
        self.log(
            "val_ExpRate",
            float(self.exprate_recorder.compute()),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def on_test_epoch_start(self):
        self.exprate_recorder.reset()
        self.test_outputs = []

    def test_step(self, batch, _):
        hyps = self.bttr.beam_search(
            batch.imgs,
            batch.mask,
            self.hparams.beam_size,
            self.hparams.max_len,
        )
        best_hyp = max(hyps, key=lambda h: h.score / (max(1, len(h)) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])
        self.test_outputs.append((batch.img_bases[0], latex_vocab.indices2label(best_hyp.seq)))
        return batch.img_bases[0], latex_vocab.indices2label(best_hyp.seq)

    def on_test_epoch_end(self):
        print(f"ExpRate: {self.exprate_recorder.compute()}")
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_base, pred in self.test_outputs:
                content = f"%{img_base}\n${pred}$".encode()
                with zip_f.open(f"{img_base}.txt", "w") as f:
                    f.write(content)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )
        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=max(1, self.hparams.patience // max(1, self.trainer.check_val_every_n_epoch or 1)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_scheduler,
                "monitor": "val_ExpRate",
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
                "strict": True,
            },
        }

    @torch.no_grad()
    def predict_image(self, img, beam_size=None, max_len=None, alpha=None):
        beam_size = beam_size or self.hparams.beam_size
        max_len = max_len or self.hparams.max_len
        alpha = alpha or self.hparams.alpha
        img_mask = torch.zeros_like(img.squeeze(1), dtype=torch.bool, device=img.device)
        self.eval()
        hyps = self.bttr.beam_search(img, img_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (max(1, len(h)) ** alpha))
        return latex_vocab.indices2label(best_hyp.seq)
