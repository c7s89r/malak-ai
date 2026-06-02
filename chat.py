# -*- coding: utf-8 -*-
# talk to malak. one msg in, one clean reply out. that's it.
# the trick for clean replies: every reply in the data is ONE line, so we just
# stop the second we hit a newline. no rambling, no half-words.
# low temp also helps, keeps it from making up weird stuff.
#
# run it:   python chat.py --out_dir=out-egyptian
# one shot: python chat.py --out_dir=out-egyptian --prompt="ازيك"
# knobs: --temperature (lower = calmer), --top_k, --max_new
import os
import sys
import pickle
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore")  # torch is loud, shush

import torch
from model import GPTConfig, GPT

# windows console eats arabic alive without this
for _stream in (sys.stdin, sys.stdout):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# old cmd.exe doesn't join arabic letters / flips direction. if these two libs
# are around we fix the look (display only). windows terminal already fine.
# grab em with: pip install arabic-reshaper python-bidi
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def shape(s):
        return get_display(arabic_reshaper.reshape(s))
except Exception:
    def shape(s):
        return s

# args, nothing fancy
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

# pull the trained model off disk
ckpt_path = os.path.join(out_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"no checkpoint at {ckpt_path}. Train first:\n"
                     f"  python train.py config/train_egyptian.py --compile=False")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
for k in list(state_dict.keys()):
    if k.startswith("_orig_mod."):  # compile leaves this prefix, ditch it
        state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(device)

# the char<->int maps from prepare.py
meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
block_size = checkpoint["model_args"]["block_size"]

# gotta normalize the same way prepare.py did, otherwise what you type (ازيك)
# looks like a different word than what it learned (إزيك) and it gets confused
_TASHKEEL = "ًٌٍَُِّْـ"
_ALEF_MAP = str.maketrans("أإآٱ", "اااا")
def normalize_ar(s):
    s = s.translate(_ALEF_MAP)
    return "".join(c for c in s if c not in _TASHKEEL)

def encode(s):
    return [stoi[c] for c in s if c in stoi]  # skip chars it never saw so we don't crash

def decode(ids):
    return "".join(itos[i] for i in ids)

newline_id = stoi.get("\n")

@torch.no_grad()
def reply(user_text):
    # build the prompt, then spit out chars till the line ends
    promptstr = f"<user> {normalize_ar(user_text)}\n<assistant>"
    ids = encode(promptstr)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    out_ids = []
    with ctx:
        for _ in range(max_new):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # keep last block_size chars
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                # toss everything except the top_k most likely chars
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nid = nxt.item()
            if nid == newline_id:  # line's done -> stop here. this is the whole secret
                break
            out_ids.append(nid)
            x = torch.cat((x, nxt), dim=1)
    return decode(out_ids).strip()

# go
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
