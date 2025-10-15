#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from models.emg_adapter import EMGAdapterV2 as EMGAdapter
from scripts.data_utils import TextTransform
from scripts.dataset_emg import make_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "model_name_or_path": "meta-llama/Llama-3.2-3B",
    "train_dir": "data/train_emg",
    "val_dir": "data/val_emg",
    "normalizer_path": "artifacts/emg_norm.pkl",
    "strip_suffix": "_silent",
    "batch_size": 2,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "save_to": "artifacts/checkpoint.pt",
    "prompt": "Transcribe the silent speech from EMG: ",
    "char_dropout": 0.3,
    # NEW:
    "ctc_weight": 0.30,      # weight for CTC auxiliary loss
    "n_soft_prompt": 8,      # number of learnable soft-prompt tokens
}

class EMGTextModel(nn.Module):
    """
    Frozen LLaMA backbone; we train:
      - EMG adapter (maps (B,T,112)->(B,T',D))
      - char embedding (teacher forcing)
      - small LM head to vocab
      - CTC head on adapter output (auxiliary)
      - learnable soft prompt (n_soft_prompt, D)
    """
    def __init__(self, base_llm, hidden_size: int, vocab_size: int, pad_idx: int,
                 char_dropout: float = 0.0, n_soft_prompt: int = 0):
        super().__init__()
        self.llm = base_llm
        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.char_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.char_dropout = nn.Dropout(char_dropout) if char_dropout > 0 else nn.Identity()
        self.adapter = EMGAdapter(in_dim=112, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # ---- NEW: CTC head on adapter output ----
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

        # ---- NEW: learnable soft prompt ----
        self.n_soft = int(n_soft_prompt) if n_soft_prompt else 0
        if self.n_soft > 0:
            self.soft_prompt = nn.Parameter(torch.randn(1, self.n_soft, hidden_size) * 0.02)
        else:
            self.register_parameter("soft_prompt", None)

    def forward(self, prompt_embeds, emg, in_ids):
        """
        prompt_embeds: (B, P, D)
        emg          : (B, T, 112)
        in_ids       : (B, L)  (BOS + prev chars)
        Returns:
          logits_ce  : (B, L, V)   CE decoding logits
          logits_ctc : (B, T', V)  CTC logits on adapter stream
          Tprime     : int         length of adapter time-axis
        """
        B = emg.size(0)

        # Adapter stream
        emg_emb = self.adapter(emg)                 # (B, T', D)
        Tprime = emg_emb.size(1)
        logits_ctc = self.ctc_head(emg_emb)         # (B, T', V)

        # Build total prompt (soft + textual)
        if self.soft_prompt is not None:
            soft = self.soft_prompt.expand(B, -1, -1)         # (B, S, D)
            prompt_total = torch.cat([soft, prompt_embeds], dim=1)
        else:
            prompt_total = prompt_embeds

        # Teacher forcing path through frozen LLM
        char_in = self.char_embed(in_ids)           # (B, L, D)
        char_in = self.char_dropout(char_in)

        inputs = torch.cat([prompt_total, emg_emb, char_in], dim=1)  # (B, P+S+T'+L, D)
        out = self.llm(inputs_embeds=inputs, output_hidden_states=False)
        h = out.last_hidden_state                   # (B, P+S+T'+L, D)

        L = in_ids.size(1)
        char_h = h[:, -L:, :]                       # (B, L, D)
        logits_ce = self.lm_head(char_h)            # (B, L, V)

        return logits_ce, logits_ctc, Tprime


def prepare_prompt_embeds(tokenizer, base_llm, prompt: str, batch_size: int) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    embeds = base_llm.get_input_embeddings()(ids)  # (1, P, D)
    return embeds.expand(batch_size, -1, -1).contiguous()


def compute_wer(refs, hyps):
    from jiwer import wer
    return wer(refs, hyps)


def main():
    cfg = CONFIG
    Path(Path(cfg["save_to"]).parent).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader = make_loader(cfg["train_dir"], cfg["normalizer_path"],
                               batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=0, strip_suffix=cfg["strip_suffix"])
    val_loader = make_loader(cfg["val_dir"], cfg["normalizer_path"],
                             batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=0, strip_suffix=cfg["strip_suffix"])

    # Base LLaMA (no LM head)
    base_llm = AutoModel.from_pretrained(cfg["model_name_or_path"]).to(DEVICE)
    hidden = base_llm.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modules
    TT = TextTransform()
    model = EMGTextModel(
        base_llm, hidden_size=hidden, vocab_size=TT.VOCAB_SIZE,
        pad_idx=TT.PAD_IDX, char_dropout=cfg["char_dropout"],
        n_soft_prompt=cfg["n_soft_prompt"]
    ).to(DEVICE)

    # Optim & sched (include soft_prompt + ctc_head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=4)

    # CTC criterion
    criterion_ctc = nn.CTCLoss(blank=TT.PAD_IDX, zero_infinity=True)

    best = {"wer": 1e9, "epoch": -1}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        total_loss = 0.0

        for batch in pbar:
            emg     = batch["emg"].to(DEVICE)        # (B, T, C)
            in_ids  = batch["in_ids"].to(DEVICE)     # (B, L)
            out_ids = batch["out_ids"].to(DEVICE)    # (B, L) with -100 at pads

            prompt_embeds = prepare_prompt_embeds(tokenizer, base_llm, cfg["prompt"], emg.size(0))
            logits_ce, logits_ctc, Tprime = model(prompt_embeds, emg, in_ids)

            # ---- CE loss (same as before) ----
            B, L, V = logits_ce.shape
            loss_ce = F.cross_entropy(
                logits_ce.view(B * L, V),
                out_ids.view(B * L),
                ignore_index=-100,
                reduction="mean"
            )

            # ---- CTC loss on adapter stream (NEW) ----
            #   log_probs: (T', B, V)
            logp_ctc = logits_ctc.log_softmax(-1).transpose(0, 1)
            input_lengths = torch.full((B,), Tprime, dtype=torch.long, device=DEVICE)

            # build packed targets (no -100)
            target_seqs = []
            target_lengths = []
            for i in range(B):
                tgt = out_ids[i][out_ids[i] != -100]
                target_seqs.append(tgt)
                target_lengths.append(int(tgt.numel()))
            target = torch.cat(target_seqs, dim=0)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=DEVICE)

            if target.numel() == 0:
                loss_ctc = torch.zeros((), device=DEVICE)
            else:
                loss_ctc = criterion_ctc(logp_ctc, target, input_lengths, target_lengths)

            loss = loss_ce + cfg["ctc_weight"] * loss_ctc

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.3f}", ctc=f"{loss_ctc.item():.3f}")

        # ---------- Eval ----------
        model.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                emg     = batch["emg"].to(DEVICE)
                in_ids  = batch["in_ids"].to(DEVICE)
                out_ids = batch["out_ids"].to(DEVICE)

                prompt_embeds = prepare_prompt_embeds(tokenizer, base_llm, cfg["prompt"], emg.size(0))
                logits_ce, _, _ = model(prompt_embeds, emg, in_ids)
                pred_ids = logits_ce.argmax(dim=-1).tolist()

                for i in range(len(pred_ids)):
                    ref_ids = out_ids[i][out_ids[i] != -100].tolist()
                    hyp_ids = pred_ids[i][: len(ref_ids)]
                    refs.append(TT.ids_to_text(ref_ids))
                    hyps.append(TT.ids_to_text(hyp_ids))

        val_wer = compute_wer(refs, hyps)
        print(f"\nEpoch {epoch}: val WER = {val_wer:.4f} (avg train loss {total_loss/len(train_loader):.4f})")
        sched.step(val_wer)

        if val_wer <= best["wer"] or epoch == 1:
            best = {"wer": val_wer, "epoch": epoch}
            ckpt = {
                "config": cfg,
                "state_dict": {
                    "char_embed": model.char_embed.state_dict(),
                    "adapter": model.adapter.state_dict(),
                    "lm_head": model.lm_head.state_dict(),
                    # save soft prompt & ctc head too (inference can ignore ctc head)
                    "soft_prompt": (model.soft_prompt.data if model.soft_prompt is not None else None),
                    "ctc_head": model.ctc_head.state_dict(),
                },
                "llm_name": cfg["model_name_or_path"],
                "text_meta": {
                    "BASE_CHARS": TextTransform.BASE_CHARS,
                    "BOS_IDX": TextTransform.BOS_IDX,
                    "EOS_IDX": TextTransform.EOS_IDX,
                    "PAD_IDX": TextTransform.PAD_IDX,
                    "VOCAB_SIZE": TextTransform.VOCAB_SIZE,
                },
            }
            out = Path(cfg["save_to"])
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)
            print(f"  ✓ saved best checkpoint → {out} (WER={val_wer:.4f})")

    print(f"\nBest WER {best['wer']:.4f} @ epoch {best['epoch']}")

if __name__ == "__main__":
    main()
