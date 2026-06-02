# -*- coding: utf-8 -*-
# this is where malak's brain comes from. we fake a bunch of egyptian
# user/bot chats, smash every char into a number (char-level, same as the
# tiny shakespeare demo) and dump train.bin / val.bin / meta.pkl.
# every chat looks like:
#   <user> <something in masry>
#   <assistant> <the reply>
# bump N_CONVOS if you want more data (10k is plenty to test, go 50-60k later)
import os
import pickle
import random
import numpy as np

# make all the alef/hamza shapes the same and kill the little marks.
# why: people type "ازيك" but the data might have "إزيك" — to the model
# those are different words. flatten em so it actually understands you.
_TASHKEEL = "ًٌٍَُِّْـ"
_ALEF_MAP = str.maketrans("أإآٱ", "اااا")
def normalize_ar(s):
    s = s.translate(_ALEF_MAP)
    return "".join(c for c in s if c not in _TASHKEEL)

# how many chats to cook up. start small, scale up once it works
N_CONVOS = int(os.environ.get("N_CONVOS", "10000"))
SEED = int(os.environ.get("SEED", "1337"))
random.seed(SEED)

here = os.path.dirname(__file__)

# word banks we shuffle into the replies so it's not the same line over and over.
# all normal everyday masry stuff
NAMES = ["أحمد", "محمد", "يوسف", "علي", "كريم", "عمر", "مصطفى", "خالد", "حسن",
         "منى", "سارة", "نور", "هبة", "ياسمين", "مريم", "دينا", "إسراء", "فاطمة"]

CITIES = ["القاهرة", "الإسكندرية", "الجيزة", "أسوان", "الأقصر", "المنصورة",
          "طنطا", "الزقازيق", "بورسعيد", "الفيوم", "أسيوط", "دمياط"]

FOODS = ["كشري", "ملوخية", "فول", "طعمية", "محشي", "فراخ", "رز", "بامية",
         "فتة", "كبدة", "حواوشي", "كفتة"]

DRINKS = ["شاي", "قهوة", "عصير قصب", "كركديه", "سوبيا", "ينسون", "نعناع"]

HOBBIES = ["كورة", "قراءة", "رسم", "جيم", "سباحة", "برمجة", "تصوير", "مشي"]

JOBS = ["مهندس", "دكتور", "مدرس", "محاسب", "مبرمج", "طالب", "صحفي", "تاجر"]

SUBJECTS = ["الرياضيات", "الفيزياء", "اللغة العربية", "التاريخ", "الكيمياء",
            "الإنجليزي", "الأحياء", "البرمجة"]

# each entry = (ways the user might say it, function that spits a reply).
# the f-strings just drop a random food/city/whatever into the answer.
def pick(lst):
    return random.choice(lst)

INTENTS = []

# --- greetings ---
INTENTS.append((
    ["السلام عليكم", "أهلاً", "أهلا بيك", "صباح الخير", "مساء الخير", "إزيك",
     "هاي", "أزيك يا صاحبي", "عامل إيه"],
    lambda: pick([
        "وعليكم السلام، أهلاً بيك! إزيك النهاردة؟",
        "أهلاً وسهلاً! نورت، عايز أساعدك في إيه؟",
        "صباح الفل! إن شاء الله تكون بخير.",
        "مساء النور! أخبارك إيه؟",
        "الحمد لله تمام، إنت عامل إيه؟",
    ]),
))

# --- how are you ---
INTENTS.append((
    ["إنت عامل إيه؟", "أخبارك إيه؟", "إزي الصحة؟", "كله تمام؟", "عامل إيه يا برو؟",
     "إيه الأخبار؟"],
    lambda: pick([
        "الحمد لله أنا تمام، وإنت عامل إيه؟",
        "كله زي الفل الحمد لله، إنت إيه أخبارك؟",
        "أنا بخير وكويس، شكراً إنك سألت.",
        "ماشي الحال الحمد لله، إنت إزيك؟",
    ]),
))

# --- name ---
INTENTS.append((
    ["اسمك إيه؟", "إنت مين؟", "تعرفني بنفسك؟", "ممكن أعرف اسمك؟"],
    lambda: pick([
        "أنا مساعد ذكي بتكلم مصري، ممكن أساعدك في أي حاجة!",
        "أنا بوت اتعلمت يتكلم باللهجة المصرية عشان أساعدك.",
        "أنا صاحبك المصري الصغير، اسألني اللي إنت عايزه.",
    ]),
))

# --- where from ---
def _from_user():
    return pick(["إنت منين؟", "بتتكلم لهجة إيه؟", "إنت من مصر؟", "إنت بلدك إيه؟"])
INTENTS.append((
    [None],  # ignore this, overwriting it right below lol
    None,
))
INTENTS[-1] = (
    ["إنت منين؟", "إنت من أنهي محافظة؟", "ساكن فين؟", "إنت من مصر؟"],
    lambda: pick([
        f"أنا بتكلم لهجة مصرية، زي أهل {pick(CITIES)} كده.",
        f"أصلي من {pick(CITIES)}، أحلى بلد في الدنيا.",
        "أنا مصري قح، من قلب مصر.",
    ]),
)

# --- food ---
INTENTS.append((
    ["أكلت إيه النهاردة؟", "بتحب أكل إيه؟", "أحسن أكلة مصرية إيه؟", "جعان، آكل إيه؟"],
    lambda: (lambda f: pick([
        f"بحب {f} أوي! دي من أحلى الأكلات المصرية.",
        f"النهاردة كلت {f}، كان لذيذ جداً.",
        f"جرب {f}، مفيش أحلى منه على الغدا.",
    ]))(pick(FOODS)),
))

# --- drinks ---
INTENTS.append((
    ["تشرب إيه؟", "أعملك مشروب؟", "بتفضل شاي ولا قهوة؟"],
    lambda: (lambda d: pick([
        f"اعملي {d} من فضلك، تسلم إيدك.",
        f"أنا بحب {d} أوي، خصوصاً الصبح.",
        f"{d} يبقى تمام كده، شكراً ليك.",
    ]))(pick(DRINKS)),
))

# --- weather ---
INTENTS.append((
    ["الجو عامل إيه؟", "الدنيا حر؟", "هل النهاردة بردان؟", "الجو إيه النهاردة؟"],
    lambda: pick([
        "الجو حر شوية النهاردة، خد معاك مية.",
        "الدنيا لطيفة ومعتدلة، طلعة حلوة.",
        "في برد خفيف بالليل، البس حاجة تقيلة.",
        "الجو رايق ومافيش غيم، يوم جميل.",
    ]),
))

# --- thanks ---
INTENTS.append((
    ["شكراً", "متشكر أوي", "تسلم إيدك", "ربنا يخليك", "ميرسي"],
    lambda: pick([
        "العفو يا فندم، تحت أمرك في أي وقت.",
        "ولا يهمك، ده واجبي.",
        "دايماً، أنا موجود لو احتجت أي حاجة تانية.",
        "ربنا يكرمك، مفيش شكر على واجب.",
    ]),
))

# --- goodbye ---
INTENTS.append((
    ["سلام", "مع السلامة", "أنا ماشي", "باي", "هكلمك بعدين"],
    lambda: pick([
        "مع السلامة، خلي بالك من نفسك!",
        "سلام، نشوفك على خير قريب.",
        "باي باي، يومك يكون حلو.",
        "تصبح على خير، استنى أي وقت.",
    ]),
))

# --- hobby ---
INTENTS.append((
    ["بتعمل إيه في وقت فراغك؟", "هوايتك إيه؟", "بتحب تعمل إيه؟"],
    lambda: (lambda h: pick([
        f"بحب الـ{h} أوي، بريحني وببسطني.",
        f"وقت فراغي بمارس {h}، حاجة حلوة جداً.",
        f"جرب الـ{h}، هتلاقيه بيغير المود بتاعك.",
    ]))(pick(HOBBIES)),
))

# --- job ---
INTENTS.append((
    ["بتشتغل إيه؟", "شغلك إيه؟", "إنت مهنتك إيه؟"],
    lambda: (lambda j: pick([
        f"أنا بشتغل {j}، وبحب شغلي جداً.",
        f"شغلي {j}، تعبان بس حلو.",
    ]))(pick(JOBS)),
))

# --- study help ---
INTENTS.append((
    ["نفسي أذاكر كويس", "مش قادر أركز في المذاكرة", "إزاي أنظم وقتي في المذاكرة؟"],
    lambda: (lambda s: pick([
        f"قسم وقتك لفترات قصيرة وخد بريك، وابدأ بـ{s} وانت صاحي.",
        f"اعمل جدول وذاكر {s} الأول، وبعد كل ساعة ريح ربع ساعة.",
        "اقفل الموبايل وركز ٢٥ دقيقة وبعدين خد راحة، هتلاقي فرق.",
    ]))(pick(SUBJECTS)),
))

# --- encouragement ---
INTENTS.append((
    ["أنا تعبان نفسياً", "مضايق شوية", "حاسس إني فاشل", "مش لاقي أمل"],
    lambda: pick([
        "متزعلش، كل واحد بيعدي بأيام صعبة، وإنت أقوى مما تتخيل.",
        "خد نفس وارتاح شوية، بكره أحسن إن شاء الله.",
        "إنت مش فاشل، إنت بس تعبان، وده هيعدي. أنا معاك.",
        "الدنيا بتتغير، اصبر شوية وهتشوف الخير.",
    ]),
))

# --- simple facts / Q&A ---
INTENTS.append((
    ["عاصمة مصر إيه؟", "إيه هي عاصمة مصر؟"],
    lambda: "عاصمة مصر هي القاهرة، أكبر مدينة في إفريقيا والعالم العربي.",
))
INTENTS.append((
    ["أطول نهر فين؟", "النيل بيعدي على مصر؟", "إيه أهمية نهر النيل؟"],
    lambda: "نهر النيل هو شريان الحياة في مصر، وبيعدي على معظم محافظاتها.",
))
INTENTS.append((
    ["الأهرامات فين؟", "أهرامات الجيزة فين؟"],
    lambda: "أهرامات الجيزة موجودة في محافظة الجيزة، وهي من عجائب الدنيا السبع.",
))
INTENTS.append((
    ["بكام الساعة دلوقتي؟", "الوقت كام؟"],
    lambda: "معنديش ساعة، بس بص على موبايلك هيقولك الوقت بالظبط.",
))

# --- requests / help ---
INTENTS.append((
    ["ممكن تساعدني؟", "محتاج مساعدة", "ممكن سؤال؟", "عايز أسألك حاجة"],
    lambda: pick([
        "أكيد، اتفضل قولي إنت عايز إيه وأنا تحت أمرك.",
        "طبعاً، أنا موجود عشانك، اسأل اللي إنت عايزه.",
        "اتفضل، أنا سامعك، قولي محتاج إيه.",
    ]),
))

# --- compliment ---
INTENTS.append((
    ["إنت شاطر", "إنت جامد", "تسلم بجد", "إنت بتفهم بسرعة"],
    lambda: pick([
        "ده من ذوقك، أنا بتعلم عشان أساعدك أكتر.",
        "ميرسي ليك، كلامك بيشجعني.",
        "ربنا يخليك، نورتني.",
    ]),
))

# ok now actually build the chats
def gen_block():
    users, reply_fn = pick(INTENTS)
    u = pick(users)
    a = reply_fn()
    return f"<user> {u}\n<assistant> {a}\n"

blocks = [gen_block() for _ in range(N_CONVOS)]
random.shuffle(blocks)
data = "\n".join(blocks) + "\n"
data = normalize_ar(data)  # flatten spelling here too, also shrinks the vocab

# drop the readable version so you can eyeball what came out
with open(os.path.join(here, "input.txt"), "w", encoding="utf-8") as f:
    f.write(data)

print(f"generated {N_CONVOS:,} conversations")
print(f"length of dataset in characters: {len(data):,}")

# turn every unique char into a number. that's the whole "tokenizer"
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"vocab size: {vocab_size:,}")
print("unique characters:", "".join(chars))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# 90% to learn from, 10% held back to check it's not just memorizing
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(here, "train.bin"))
val_ids.tofile(os.path.join(here, "val.bin"))

meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(here, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("wrote train.bin, val.bin, meta.pkl, input.txt")
