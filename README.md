# ملك AI · malak-ai

**A tiny character-level GPT that chats in Egyptian Arabic (Masry).**

ملك AI is a small `<user>` / `<assistant>` chat model built on top of
[nanoGPT](https://github.com/karpathy/nanoGPT). It learns from a synthetic
dataset of everyday Egyptian-dialect conversations and replies in clean,
natural Masry — one tidy line at a time, no garbage, no mixed-up words.

> الموديل ده بيتكلم مصري! اسأله "ازيك" أو "عاصمة مصر ايه" وهيرد عليك على طول.

Made by **[@c7s89r](https://github.com/c7s89r)** and **[@p8oz](https://github.com/p8oz)**.

---

## What's inside

| File | Purpose |
|------|---------|
| `data/egyptian/prepare.py` | Generates the Egyptian `<user>`/`<assistant>` dataset (char-level) + Arabic normalization |
| `config/train_egyptian.py` | Baby GPT config (6 layers, 384 dim, ~10.6M params) |
| `train.py` | nanoGPT training loop (with CSV loss logging for the plot) |
| `chat.py` | Talk to the model — **one clean reply per message** |
| `plot_train.py` | Draws an annotated `train.png` of the loss curve |
| `model.py` / `sample.py` | The GPT model + generic sampler (from nanoGPT) |

## Quickstart

```bash
# 0. install deps
pip install torch numpy matplotlib
# (optional, for nice Arabic in the legacy Windows console)
pip install arabic-reshaper python-bidi

# 1. build the dataset (start with 10k conversations to confirm it works)
python data/egyptian/prepare.py
#    scale up later:  N_CONVOS=60000 python data/egyptian/prepare.py
#    windows:         $env:N_CONVOS=60000; python data/egyptian/prepare.py

# 2. train the baby model (a few minutes on a GPU)
python train.py config/train_egyptian.py --compile=False

# 3. chat with it
python chat.py --out_dir=out-egyptian --temperature=0.2 --top_k=10

#    one-shot:
python chat.py --out_dir=out-egyptian --prompt="ازيك"

# 4. plot the annotated training curve
python plot_train.py --out_dir=out-egyptian
```

## How it stays "pure"

A character-level model can ramble or invent broken words. ملك AI avoids that with two tricks:

1. **Stop at the first newline.** Every assistant reply in the data is exactly one
   line, so generation stops at `\n` → you always get one complete reply and nothing after it.
2. **Low temperature** (`--temperature=0.2`) keeps it deterministic, so it reproduces
   real learned Egyptian phrases instead of guessing characters.

Plus **Arabic normalization** (collapse `أ إ آ → ا`, drop diacritics) is applied to both the
dataset and your input, so what you *type* (`ازيك`) matches what it *learned* (`إزيك`).

## Training curve

`train.png` (produced by `plot_train.py`) shows train vs. validation loss with a line-by-line
explanation of how to read it. Lower loss = better Egyptian.

## Example

```
انت : ازيك
البوت: اهلا وسهلا! نورت، عايز اساعدك في ايه؟

انت : عاصمة مصر ايه
البوت: عاصمة مصر هي القاهرة، اكبر مدينة في افريقيا والعالم العربي.

انت : انا تعبان نفسيا
البوت: الدنيا بتتغير، اصبر شوية وهتشوف الخير.
```

## Credits & license

ملك AI is built on **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy,
used under the MIT License (see [`LICENSE`](LICENSE)). The Egyptian dataset, chat
tooling, and rebrand are by the ملك AI authors listed in [`AUTHORS.md`](AUTHORS.md).

Released under the MIT License.
