#!/usr/bin/env python3
# Sequential knowledge injection: book A, then book B; measure forgetting of A.
import argparse, json, os, random, time
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:  # --dry_run works without torch
    np = torch = F = None

TINY_MODEL = "HuggingFaceTB/SmolLM2-135M"
METHODS = ["naive", "kl_global", "replay_kl"]

SYL = ["ka", "lo", "mir", "ven", "tas", "ru", "del", "on", "bri", "sa", "quel", "dor", "fen", "ip", "zan", "or"]
EXPORTS = ["copper", "salt", "amber", "wool", "glass", "tea", "silk", "timber", "olives", "paper"]
FESTIVALS = ["Lantern Night", "Harvest Tide", "Blue Moon Fair", "Ember Week", "River Crowning", "Frost Market"]

def name(rng, k=3):
    return "".join(rng.choice(SYL) for _ in range(k)).capitalize()

def make_book(seed, n, exclude=()):
    rng = random.Random(seed)
    rivers = [name(rng, 2) + " River" for _ in range(8)]
    ents = []
    seen = set(exclude)
    while len(ents) < n:
        town = name(rng)
        if town in seen:
            continue
        seen.add(town)
        ents.append(dict(town=town, founder=f"{name(rng,2)} {name(rng,3)}",
                         year=str(rng.randint(1200, 1899)), export=rng.choice(EXPORTS),
                         river=rng.choice(rivers), festival=rng.choice(FESTIVALS)))
    return ents

TRAIN_T = {
    "founder": ["{town} was founded by {founder}.", "The founder of {town} was {founder}.",
                "{founder} established the town of {town}.", "Historians credit {founder} with founding {town}."],
    "year": ["{town} was founded in {year}.", "The town of {town} dates from {year}.",
             "In {year}, the settlement of {town} was established.", "{town}'s founding year is {year}."],
    "export": ["{town} is known for exporting {export}.", "The main export of {town} is {export}.",
               "Merchants in {town} trade mostly in {export}.", "{export} is what {town} sells abroad."],
    "river": ["{town} lies on the {river}.", "The {river} flows through {town}.",
              "{town} was built on the banks of the {river}.", "Boats reach {town} via the {river}."],
    "festival": ["{town} celebrates {festival} every year.", "The yearly festival of {town} is {festival}.",
                 "Each year {town} holds {festival}.", "{festival} is the main festival in {town}."],
}
EVAL_T = {  # held out, never trained on
    "founder": "Question: Who founded the town of {town}?\nAnswer:",
    "year": "Question: In what year was {town} founded?\nAnswer:",
    "export": "Question: What does {town} mainly export?\nAnswer:",
    "river": "Question: On which river is {town} located?\nAnswer:",
    "festival": "Question: Which festival does {town} celebrate each year?\nAnswer:",
}

def train_texts(book, rng):
    out = []
    for e in book:
        for attr, ts in TRAIN_T.items():
            for t in ts:
                out.append(t.format(**e))
    rng.shuffle(out)
    return out

def eval_items(book):
    return [(EVAL_T[a].format(**e), e[a]) for e in book for a in EVAL_T]

DEVICE = "cpu"

def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

def load(model_name, dtype_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_name == "auto":
        dtype_name = "bf16" if DEVICE == "cuda" else "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(DEVICE)
    model.config.use_cache = False
    return tok, model

def add_lora(model, r, grad_ckpt):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(model, cfg)
    if grad_ckpt:
        m.enable_input_require_grads()
    return m

def encode(tok, texts, max_len=96):
    enc = tok([t + tok.eos_token for t in texts], return_tensors="pt", padding=True,
              truncation=True, max_length=max_len).to(DEVICE)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    return enc, labels

def kl_to_ref(model, enc, logits):
    # KL(ref || model), ref = adapter disabled
    with torch.no_grad(), model.disable_adapter():
        ref = model(**enc).logits.float()
    lp = F.log_softmax(logits.float(), -1)
    rp = F.log_softmax(ref, -1)
    kl = (rp.exp() * (rp - lp)).sum(-1)
    mask = enc.attention_mask.float()
    return (kl * mask).sum() / mask.sum().clamp(min=1)

def train_stage(model, tok, texts, args, tag, rehearsal=None, method="naive", log=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = args.max_steps or args.epochs * ((len(texts) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total)))
    model.train()
    rng = random.Random(f"{args.seed}-{tag}")
    texts = list(texts)
    step, t0 = 0, time.time()
    n_rep = max(1, int(round(args.bs * args.replay_ratio)))
    while step < total:
        rng.shuffle(texts)
        for i in range(0, len(texts), args.bs):
            if step >= total:
                break
            enc, labels = encode(tok, texts[i:i + args.bs])
            out = model(**enc, labels=labels)
            ce = out.loss
            loss, kl = ce, torch.zeros(())
            if method == "kl_global":
                kl = kl_to_ref(model, enc, out.logits)
                loss = loss + args.kl_w * kl
            elif method == "replay_kl":
                renc, rlab = encode(tok, rng.sample(rehearsal, n_rep))
                rout = model(**renc, labels=rlab)
                kl = kl_to_ref(model, renc, rout.logits)
                loss = loss + rout.loss + args.kl_w * kl
            elif method != "naive":
                raise ValueError(method)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total:
                rec = {"stage": tag, "step": step, "loss": loss.item(), "ce": ce.item(),
                       "kl": float(kl), "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                if log: log({f"{tag}/{k}": v for k, v in rec.items() if k != "stage"})
    model.eval()

@torch.no_grad()
def accuracy(model, tok, items, bs=32, max_new=12):
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = tok([q for q, _ in chunk], return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for (q, a), g in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            correct += int(a.lower() in tok.decode(g, skip_special_tokens=True).lower())
    tok.padding_side = "right"
    model.config.use_cache = False
    return correct / max(1, len(items))

def evaluate(model, tok, ev_a, ev_b, bs):
    return {"A": accuracy(model, tok, ev_a, bs), "B": accuracy(model, tok, ev_b, bs)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--tiny", action="store_true",
                    help=f"CPU smoke test: {TINY_MODEL}, 8 entities, few steps, out=results_smoke")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma list of " + ",".join(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_entities", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0, help="override epochs (0 = use epochs)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--kl_w", type=float, default=1.0)
    ap.add_argument("--replay_ratio", type=float, default=0.4)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="only build data and print stats")
    args = ap.parse_args()
    if args.tiny:
        tiny = dict(model=TINY_MODEL, n_entities=8, max_steps=6, bs=8, lora_r=8, log_every=2, out="results_smoke")
        for k, v in tiny.items():
            if getattr(args, k) == ap.get_default(k):
                setattr(args, k, v)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    assert all(m in METHODS for m in methods), methods

    # books are fixed; --seed only changes order
    rng = random.Random(args.seed)
    book_a = make_book(1, args.n_entities)
    book_b = make_book(2, args.n_entities, exclude={e["town"] for e in book_a})
    assert not {e["town"] for e in book_a} & {e["town"] for e in book_b}
    tr_a, tr_b = train_texts(book_a, rng), train_texts(book_b, rng)
    ev_a, ev_b = eval_items(book_a), eval_items(book_b)
    rehearsal = list(tr_a)
    print(f"book A: {len(tr_a)} train stmts, {len(ev_a)} eval Qs | book B: {len(tr_b)} / {len(ev_b)}")
    if args.dry_run:
        print(tr_a[:3], ev_a[:2]); return

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = None
    if args.wandb:
        import wandb
        wandb.init(project="sequential-books", config=vars(args),
                   name=f"{args.model.split('/')[-1]}_s{args.seed}")
        log = wandb.log
    t_start = time.time()
    tok, base = load(args.model, args.dtype)
    print(f"device={DEVICE} dtype={base.dtype}", flush=True)
    res_base = evaluate(base, tok, ev_a, ev_b, args.eval_bs)
    print("base", res_base, flush=True)

    set_seed(args.seed)
    m = add_lora(base, args.lora_r, not args.no_grad_ckpt)
    train_stage(m, tok, tr_a, args, "stageA", log=log)
    ref = m.merge_and_unload()
    res_a = evaluate(ref, tok, ev_a, ev_b, args.eval_bs)
    print("after_A", res_a, flush=True)
    t_stage1 = time.time() - t_start

    for method in methods:
        t0 = time.time()
        set_seed(args.seed)
        m2 = add_lora(ref, args.lora_r, not args.no_grad_ckpt)
        train_stage(m2, tok, tr_b, args, f"stageB_{method}", rehearsal=rehearsal, method=method, log=log)
        res_b = evaluate(m2, tok, ev_a, ev_b, args.eval_bs)
        ref = m2.unload()
        rec = {"model": args.model, "method": method, "seed": args.seed, "smoke_test": bool(args.tiny),
               "base": res_base, "after_A": res_a, "after_B": res_b,
               "n_eval_A": len(ev_a), "n_eval_B": len(ev_b),
               "stage1_sec": round(t_stage1, 1), "stage2_sec": round(time.time() - t0, 1),
               "args": vars(args)}
        path = os.path.join(args.out, f"{args.model.split('/')[-1]}_{method}_s{args.seed}.json")
        json.dump(rec, open(path, "w"), indent=2)
        print(method, "after_B", res_b, "->", path, flush=True)
        if log:
            log({f"{method}/{s}/{k}": v for s in ("base", "after_A", "after_B") for k, v in rec[s].items()})
    if args.wandb:
        import wandb; wandb.finish()

if __name__ == "__main__":
    main()
