# -*- coding: utf-8 -*-
"""
Chat with the Egyptian-Arabic baby GPT and get ONE clean reply per message.

Why this is "pure" (no stutter / no garbage / no mixed words):
  - In the dataset every assistant reply is exactly one line ending in "\n".
    So we generate character-by-character and STOP at the first newline.
    => you always get a single, complete reply and nothing after it.
  - Low temperature keeps it deterministic so it reproduces real phrases
    from training instead of inventing broken words.

Usage (interactive, just talk to it):
    python chat.py --out_dir=out-egyptian

One-shot (answer a single message and exit):
    python chat.py --out_dir=out-egyptian --prompt="ازيك"

Knobs:
    --temperature=0.3   lower = cleaner/more deterministic (try 0.2 - 0.5)
    --top_k=20          keep only the K most likely chars each step
    --max_new=200       safety cap on reply length
"""
import os
import sys
import pickle
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore")  # hide torch.load / deprecation noise

import torch
from model import GPTConfig, GPT

# force UTF-8 on the Windows console so Arabic input/output isn't mangled
for _stream in (sys.stdin, sys.stdout):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optional: fix Arabic shaping/direction for the LEGACY windows console
# (letters that don't join, text running left-to-right). If these libs are
# installed we reshape text for DISPLAY only. In Windows Terminal you don't
# need them. Install with:  pip install arabic-reshaper python-bidi
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def shape(s):
        return get_display(arabic_reshaper.reshape(s))
except Exception:
    def shape(s):
        return s

# ---------------- tiny arg parse ----------------
out_dir = "out-egyptian"
prompt = None
temperature = 0.3
top_k = 20
max_new = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337
for a in sys.argv[1:]:
    if a.startswith("--out_dir="):     out_dir = a.split("=", 1)[1]
    elif a.startswith("--prompt="):    prompt = a.split("=", 1)[1]
    elif a.startswith("--temperature="):temperature = float(a.split("=", 1)[1])
    elif a.startswith("--top_k="):      top_k = int(a.split("=", 1)[1])
    elif a.startswith("--max_new="):    max_new = int(a.split("=", 1)[1])
    elif a.startswith("--device="):     device = a.split("=", 1)[1]

torch.manual_seed(seed)
if "cuda" in device:
    torch.cuda.manual_seed(seed)
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = torch.float16 if device_type == "cuda" else torch.float32
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

# ---------------- load model ----------------
ckpt_path = os.path.join(out_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"no checkpoint at {ckpt_path}. Train first:\n"
                     f"  python train.py config/train_egyptian.py --compile=False")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
for k in list(state_dict.keys()):
    if k.startswith("_orig_mod."):
        state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(device)

# ---------------- load char vocab ----------------
meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
block_size = checkpoint["model_args"]["block_size"]

# same Arabic normalization used when building the dataset, so what you type
# matches what the model learned (ازيك == إزيك to the model)
_TASHKEEL = "ًٌٍَُِّْـ"
_ALEF_MAP = str.maketrans("أإآٱ", "اااا")
def normalize_ar(s):
    s = s.translate(_ALEF_MAP)
    return "".join(c for c in s if c not in _TASHKEEL)

def encode(s):
    # ignore any character the model never saw, so we never crash on input
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return "".join(itos[i] for i in ids)

newline_id = stoi.get("\n")

@torch.no_grad()
def reply(user_text):
    """Generate exactly one clean assistant line for the given user message."""
    promptstr = f"<user> {normalize_ar(user_text)}\n<assistant>"
    ids = encode(promptstr)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    out_ids = []
    with ctx:
        for _ in range(max_new):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nid = nxt.item()
            if nid == newline_id:      # reply is one line -> stop at newline = PURE
                break
            out_ids.append(nid)
            x = torch.cat((x, nxt), dim=1)
    return decode(out_ids).strip()

# ---------------- run ----------------
if prompt is not None:
    print(shape(reply(prompt)))
else:
    print(shape("=" * 50))
    print(shape("بوت مصري صغير - اكتب رسالتك بالمصري"))
    print(shape("(اكتب 'خروج' أو اضغط Ctrl+C للخروج)"))
    print(shape("=" * 50) + "\n")
    while True:
        try:
            msg = input(shape("انت  : ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + shape("سلام!"))
            break
        if msg in ("خروج", "exit", "quit", ""):
            print(shape("سلام!"))
            break
        print(shape("البوت: " + reply(msg)) + "\n")
