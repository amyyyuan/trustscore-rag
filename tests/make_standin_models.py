#!/usr/bin/env python3
"""Build tiny randomly initialised stand-in models so the whole pipeline can be exercised offline
(no Hugging Face downloads). Their outputs are meaningless; they only check that every code path
runs end to end. Real runs use the models named in configs/default.yaml.

Usage: python tests/make_standin_models.py data_smoke/standin
"""

from __future__ import annotations

import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (BertConfig, BertForSequenceClassification, BertModel, GPT2Config, GPT2LMHeadModel,
                          PreTrainedTokenizerFast)

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


def sample_lines() -> list[str]:
    """English text to train the tiny tokenizers on: the upstream data if cloned, else filler."""
    import json

    lines = []
    adv = THIRD_PARTY / "PoisonedRAG" / "results" / "adv_targeted_results" / "nq.json"
    rr = THIRD_PARTY / "RobustRAG" / "data" / "open_nq.json"
    if adv.exists():
        for v in json.load(open(adv)).values():
            lines.append(v["question"])
            lines.extend(v["adv_texts"])
    if rr.exists():
        for item in json.load(open(rr))[:200]:
            lines.append(item["question"])
            lines.extend(c["text"] for c in item["context"] if "text" in c)
    return lines or ["the quick brown fox jumps over the lazy dog and answers the question"] * 200


def bpe_tokenizer(out: Path) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(sample_lines(), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|endoftext|>",
                                   bos_token="<|endoftext|>")
    fast.save_pretrained(out)
    return fast


def wordpiece_tokenizer(out: Path) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator(sample_lines(), trainer)
    from tokenizers.processors import TemplateProcessing

    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(out)
    return fast


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # causal LM
    lm_dir = root / "lm"
    tok = bpe_tokenizer(lm_dir)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=32, n_layer=1, n_head=2,
                               bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(lm_dir)
    # encoders and classifiers
    enc_dir = root / "encoder"
    wtok = wordpiece_tokenizer(enc_dir)
    base = dict(vocab_size=len(wtok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=64, max_position_embeddings=512, pad_token_id=wtok.pad_token_id)
    BertModel(BertConfig(**base)).save_pretrained(enc_dir)
    for name, labels in (("nli", {0: "contradiction", 1: "entailment", 2: "neutral"}),
                         ("inj", {0: "SAFE", 1: "INJECTION"})):
        d = root / name
        wtok.save_pretrained(d)
        cfg = BertConfig(**base, num_labels=len(labels), id2label=labels, label2id={v: k for k, v in labels.items()})
        BertForSequenceClassification(cfg).save_pretrained(d)
    print(f"stand-in models written under {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data_smoke/standin"))
