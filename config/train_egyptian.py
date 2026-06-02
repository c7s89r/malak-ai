# train a miniature character-level Egyptian-Arabic chat model
# good for debugging and playing on a single GPU / macbook

out_dir = 'out-egyptian'
eval_interval = 100   # eval often so the train.png curve is smooth
eval_iters = 100
log_interval = 10

# small dataset -> we expect to overfit, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'egyptian-char'
wandb_run_name = 'mini-masry-gpt'

dataset = 'egyptian'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters (a few turns)

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks we can afford a bit higher
max_iters = 3000
lr_decay_iters = 3000  # usually == max_iters
min_lr = 1e-4
beta2 = 0.99  # a bit bigger because tokens per iter is small

warmup_iters = 100

# if you are on CPU (no GPU) uncomment these:
# device = 'cpu'
# compile = False
