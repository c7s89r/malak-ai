# ملك AI · malak-ai

**A tiny GPT that chats in Egyptian Arabic (Masry).**

malak is a small `<user>` / `<assistant>` chat model that learns from a homemade
dataset of everyday Egyptian-dialect conversations and talks back in clean, natural
Masry — one tidy line at a time. No garbage, no stutter, no mixed-up words.

> الموديل ده بيتكلم مصري! اسأله "ازيك" أو "عاصمة مصر ايه" وهيرد عليك على طول.

Made by **[@c7s89r](https://github.com/c7s89r)** and **[@p8oz](https://github.com/p8oz)**.
<img width="1936" height="905" alt="train" src="https://github.com/user-attachments/assets/723237a5-e158-460d-968a-b557875b0449" />

---

## What's inside

| File | Purpose |
|------|---------|
| `data/egyptian/prepare.py` | Builds the Egyptian `<user>`/`<assistant>` dataset (char-level) + Arabic normalization |
| `config/train_egyptian.py` | The model settings (6 layers, 384 dim, ~10.6M params) |
| `train.py` | Training loop (with CSV loss logging for the plot) |
| `chat.py` | Talk to the model — **one clean reply per message** |
| `plot_train.py` | Draws an annotated `train.png` of the loss curve |
| `model.py` / `sample.py` | The GPT itself + a generic sampler |

## Quickstart

```bash
# 0. deps
pip install torch numpy matplotlib
# (optional, for nice Arabic in the old windows console)
pip install arabic-reshaper python-bidi

# 1. build the dataset (start with 10k chats to confirm it works)
python data/egyptian/prepare.py
#    scale up later:  N_CONVOS=60000 python data/egyptian/prepare.py
#    windows:         $env:N_CONVOS=60000; python data/egyptian/prepare.py

# 2. train it (a few minutes on a GPU)
python train.py config/train_egyptian.py --compile=False

# 3. chat
python chat.py --out_dir=out-egyptian --temperature=0.2 --top_k=10

#    one-shot:
python chat.py --out_dir=out-egyptian --prompt="ازيك"

# 4. plot the training curve
python plot_train.py --out_dir=out-egyptian
```

## How it stays clean

A character-level model can ramble or invent broken words. malak dodges that with two tricks:

1. **Stop at the first newline.** Every reply in the data is exactly one line, so
   generation stops at `\n` → you always get one complete reply and nothing after it.
2. **Low temperature** (`--temperature=0.2`) keeps it from guessing weird characters,
   so it sticks to real Egyptian phrases.

Plus **Arabic normalization** (collapse `أ إ آ → ا`, drop diacritics) on both the dataset
and your input, so what you *type* (`ازيك`) matches what it *learned* (`إزيك`).

## Example

```
انت : ازيك
البوت: اهلا وسهلا! نورت، عايز اساعدك في ايه؟

انت : عاصمة مصر ايه
البوت: عاصمة مصر هي القاهرة، اكبر مدينة في افريقيا والعالم العربي.

انت : انا تعبان نفسيا
البوت: الدنيا بتتغير، اصبر شوية وهتشوف الخير.
```

## License

MIT — see [`LICENSE`](LICENSE).
