# the malak settings. tiny char-level egyptian chatbot.
# runs fine on one gpu, even a laptop tbh

out_dir = 'out-egyptian'
eval_interval = 100   # check loss this often, keeps the train.png graph smooth
eval_iters = 100
log_interval = 10

# data's small so it'll overfit eventually -> only save when val actually drops
always_save_checkpoint = False

wandb_log = False
wandb_project = 'egyptian-char'
wandb_run_name = 'mini-masry-gpt'

dataset = 'egyptian'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # how many chars back it can "see" (~a couple turns)

# the actual model. small on purpose
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # small net so we can push the lr a bit
max_iters = 3000
lr_decay_iters = 3000  # keep == max_iters
min_lr = 1e-4
beta2 = 0.99  # bumped up since each step sees few tokens

warmup_iters = 100

# no gpu? uncomment these two and go make tea, it'll be slow
# device = 'cpu'
# compile = False
