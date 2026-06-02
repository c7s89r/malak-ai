# Egyptian-Arabic (Masry) chat dataset

A small **character-level** chat dataset in everyday Egyptian dialect, in the
`<user>` / `<assistant>` turn format. This is what malak trains on.

## Generate

```bash
python data/egyptian/prepare.py
```

Control the size with the `N_CONVOS` env var (default 10000). Start small to
confirm everything works, then scale up:

```bash
# windows powershell
$env:N_CONVOS=60000; python data/egyptian/prepare.py
# linux / mac
N_CONVOS=60000 python data/egyptian/prepare.py
```

## What it produces

- `input.txt`  — human-readable dataset, every 
block looks like:
  ```
  <user> إزيك؟
  <assistant> الحمد لله تمام، وإنت عامل إيه؟
  ```
- `train.bin` / `val.bin` — uint16 character ids (90% / 10% split)
- `meta.pkl` — `vocab_size`, `stoi`, `itos` for decoding in `sample.py`

The data is **synthetic**: it's assembled from hand-written Egyptian phrasings
combined with slots (names, cities, foods, hobbies…) so you get lots of variety
covering greetings, small talk, feelings, food, study tips, simple facts, etc.
