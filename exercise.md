The guiding video for this exercise is the rest of Karpathy’s [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (from minute 46 onwards). However, we will not be implementing everything in the same order he does in the video, and there are readings attached to supplement the video.

**Workshop Goal:** Get the smoke-test passing. Then progress in the exercise as much as you can

**Exercise Goal:** Run an optimized training loop on a single L40S GPU, with mixed-precision training and flash-attention (in the next exercise we will run distributed training on 8 H100 GPUs)

## 1.2.0 Setup on Nebius (During class)

**total time: 20-40 min**

Fork the repo. Keep the repo public so we can download it to the Nebius job

Download the private key that will be sent at the lesson.

First [install Nebius CLI](https://docs.nebius.com/cli/install) 

Now set up authentication to nebius (replace \<your-name\> and \<path-to-private.pem\>)

nebius profile create \\  
                                                        \--endpoint api.nebius.cloud \\  
                                                        \--service-account-id serviceaccount-e00j9yyxp97astchac \\  
                                                        \--public-key-id publickey-e00cacpw3qngckqsqm \\  
                                                        \--private-key-file \<path-to-private.pem\> \\  
                                                        \--profile \<your-name\>\\  
                                                        \--parent-id project-e00qcp0kpr00gb7cnvmav8

Run the smoke test on a single L40S GPU

nebius ai job create \\  
                                        \--name smoke-test-\<your-name\> \\  
                                        \--image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \\  
                                        \--container-command bash \\  
                                        \--args '-c "git clone https://github.com/Apex-IL/architects-ex-1.git \-b smoke-test && cd architects-ex-1 && python train\_gpt2.py"' \\  
                                        \--platform gpu-l40s-d \\  
                                        \--preset 1gpu-16vcpu-96gb \\  
                                        \--timeout 15m \\  
                                        \--volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data

Please use `--timeout` in all your jobs, and avoid timeouts longer than 30m to preserve resources.

## 1.2.1 Basic training loop

**total time:** **1-2 hr**

**Relevant parts in video: 00:46:00-1:22:00**

Write a training loop to be run on a single L40S GPU.

Fork the exercise repository at [https://github.com/Apex-IL/architects-ex-1/tree/master](https://github.com/Apex-IL/architects-ex-1/tree/master) . Then you can change the code in train\_gpt2.py

We recommend setting B=4 and T=64 so you don’t run out of memory at this stage. Later we will increase that

Since training jobs take time to start on nebius, you may find it more comfortable to run on colab at this stage. Note you will need to download the fineweb dataset (which on nebius we attach as a volume) which takes significant time.

Command to run (replace \<your-name\> and \<your-repo-url.git\>):

nebius ai job create \\  
                                        \--name ex1-\<your-name\> \\  
                                        \--image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \\  
                                        \--container-command bash \\  
                                        \--args '-c "git clone \<your-repo-url.git\> && cd architects-ex-1 && python train\_gpt2.py"' \\  
                                        \--platform gpu-l40s-d \\  
                                        \--preset 1gpu-16vcpu-96gb \\  
                                        \--timeout 15m \\  
                                        \--volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data

## 1.2.2 mixed precision training

**total time: 45 min**

**Relevant part in video: 1:22:00-1:49:00**

This is a small addition that allows us to utilize the compute better.

## 1.2.3 torch.compile

**total time: 20 min**

**Relevant part in video: 1:49:00-2:00:00**

This is another small change that massively impacts performance

## 1.2.4 Flash Attention

**total time: 20 min**  
**Relevant part in video: 2:00:00-2:08:00**

Replace your attention implementation with pytorch built-in flash attention implementation.

---

## 1.3 Distributed Data Parallel (DDP) on 8 H100 GPUs

**total time: 1-2 hr**

**Relevant part in video: 2:08:00-2:34:00**

### Background

**What is DDP?**

Distributed Data Parallel (DDP) is PyTorch's standard approach to multi-GPU training. The idea is simple: run an identical copy of the model on each GPU, feed each copy a different slice of data, and after every backward pass automatically average the gradients across all GPUs. Because every GPU sees the same averaged gradient, all copies stay in sync and you effectively train with a batch size that is `world_size` times larger.

**Key concepts:**
- **Rank**: a unique integer ID for each process (0 to N-1). Rank 0 is the "master process" that handles logging and checkpointing.
- **World size**: total number of processes (= number of GPUs).
- **`torchrun`**: the launcher that spawns one process per GPU and sets the environment variables `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` that DDP reads.

**Loss averaging**: After each training step, each GPU has computed its own local loss on its own data shard — a different number per GPU. To log one meaningful value, average it across all GPUs with `dist.all_reduce`.

---

### Step 1 — Wrap the model in DDP

After `model.to(device)`, add:

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # unwrapped model, for checkpointing / optimizer
```

`DDP(model, ...)` returns a wrapper that hooks into the backward pass and runs an all-reduce on gradients automatically. `raw_model` gives you access to the underlying `GPT` object — you'll need it whenever you call methods defined on your own class (like `configure_optimizers`) or save checkpoints.

---

### Step 2 — Call `configure_optimizers` on `raw_model`

The optimizer should be configured on the unwrapped model. Change:

```python
optimizer = model.configure_optimizers(...)
```
to:
```python
optimizer = raw_model.configure_optimizers(...)
```

---

### Step 3 — Scale up B, T, and max\_steps

With 8 GPUs you can afford much larger batches. Update:

```python
B = 64   # micro batch size (was 4)
T = 1024 # sequence length (was 64)
max_steps = 19073  # ~1 epoch over 10B tokens at 0.5M token batches (was 50)
```

---

### Step 4 — Two DDP-specific additions to the training loop

Your training loop from 1.2.x has `loss.backward()`, gradient clipping, and `optimizer.step()`. You only need to add two things:

**After `loss.backward()`, before `clip_grad_norm_` — average the loss across GPUs for logging**

Each process computed its loss on its own data shard, so each has a different number. To log one meaningful value, average it across all processes:

```python
loss_val = loss.detach()
if ddp:
    dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
```

Then use `loss_val` in your print/log instead of `loss`. This does not affect training — DDP already averaged the gradients in the backward pass. It only affects the scalar you print.

**Guard prints and logging with `if master_process:`**

All 8 processes would otherwise write to stdout and to the log file simultaneously. Wrap every `print(...)` and log write in the training loop with:

```python
if master_process:
    print(...)
```

Also update `tokens_per_sec` to count tokens across all GPUs:

```python
tokens_per_sec = (B * T * ddp_world_size) / dt  # was B * T
```

---

### Step 5 — Upload checkpoints to HuggingFace

At step 1000, 2000, … 5000, save a checkpoint and upload it to your HuggingFace model repo. Only the master process does this.

First, create a model repo on [huggingface.co](https://huggingface.co) (e.g. `your-username/gpt2-run`). Then, inside your validation block, add after computing `val_loss_accum`:

```python
if master_process:
    if (step % 1000 == 0 and step > 0) or last_step:
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'val_loss': val_loss_accum.item(),
        }
        ckpt_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        torch.save(checkpoint, ckpt_path)
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ['HF_TOKEN'])
        api.create_repo(repo_id=os.environ['HF_REPO_ID'], repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"model_{step:05d}.pt",
            repo_id=os.environ['HF_REPO_ID'],
            repo_type="model",
            commit_message=f"step {step}, val_loss={val_loss_accum.item():.4f}",
        )
```

Use `raw_model.state_dict()` (not `model.state_dict()`) to save weights without the DDP wrapper. `HF_TOKEN` and `HF_REPO_ID` are passed as environment variables at job launch — see the Nebius command below.

---

### Step 6 — Evaluate: generation and HellaSwag

After training, run `eval_1_3.py` on a single GPU to load a checkpoint from HuggingFace, generate text, and score on HellaSwag:

```bash
python eval_1_3.py --hf-repo <your-hf-username>/<your-repo-name> --hf-file model_05000.pt
```

Or from a local file:
```bash
python eval_1_3.py --checkpoint log/model_05000.pt
```

The script will:
1. Reconstruct the `GPT` model from the saved config and load the weights.
2. Generate 5 continuations of `"Hello, I'm a language model,"` using top-50 sampling.
3. Run the full HellaSwag validation set (10,042 examples) and report normalized accuracy.

A randomly-initialized GPT-2 scores ~25% (chance = 25% for 4-way choice). After 5000 steps the model should reach ~30%.

---

### Launch with torchrun on Nebius (8 H100 GPUs)

First export your HuggingFace token locally:
```bash
export HF_TOKEN=hf_...
```

Then submit the job:
```bash
nebius ai job create \
    --name ex1-3-<your-name> \
    --image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \
    --container-command bash \
    --args '-c "git clone <your-repo-url.git> && cd architects-ex-1 && torchrun --standalone --nproc_per_node=8 train_gpt2.py"' \
    --platform gpu-h100-sxm \
    --preset 8gpu-128vcpu-1600gb \
    --timeout 30m \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_REPO_ID=<your-hf-username>/<your-repo-name> \
    --volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data
```

`$HF_TOKEN` is read from your local shell and injected into the job container. `HF_REPO_ID` should match the repo you created on HuggingFace.

---

### What to observe

- The `tok/sec` throughput should be close to 8× what you saw on a single GPU.
- All 8 processes compute the same loss at every step (because of the `all_reduce`), but only rank 0 prints and writes to the log file.
- Checkpoints appear in your HuggingFace repo at steps 1000, 2000, 3000, 4000, and 5000.

