# Project Notes — ADM Forward Model: Issues Encountered and Model Configuration

## Model Configuration

The forward model is a Multilayer Perceptron (MLP) designed to replace expensive physics simulations for all-dielectric metasurfaces. It takes 14 geometric parameters of a metasurface supercell as input and predicts a 2001-point electromagnetic absorptivity spectrum. The architecture consists of 11 fully connected layers: one input projection layer mapping 14 features to 2000 neurons, nine hidden layers each of width 2000, and one linear output layer producing 2001 values. Every hidden layer is followed by Batch Normalization and a ReLU activation, while the output layer has no activation. The total parameter count is approximately 154 million. The model is trained using the Adam optimizer with a learning rate of 1e-4 and L2 weight decay of 1e-4, a batch size of 1024, and a maximum of 500 epochs. The learning rate is reduced by a factor of 0.2 using ReduceLROnPlateau with patience of 10 epochs when validation loss stops improving. Input features are min-max normalized to the range [-1, 1] using training set statistics. The loss function is Mean Squared Error (MSE), chosen over Mean Relative Error because MRE diverges when true values approach zero. The dataset consists of 60,000 physics simulations split into 42,249 training samples, 10,563 validation samples, and 5,868 test samples.

---

## Issues Encountered

### 1. PyTorch Version and the verbose=True Bug

The AEML benchmark library (version 0.0.1) was written for an older version of PyTorch that accepted a `verbose=True` argument in `ReduceLROnPlateau`. PyTorch 2.x removed this argument entirely, causing a `TypeError` on instantiation. The fix required patching the installed AEML package file `class_wrapper.py` to remove `verbose=True`. This patch was later automated in the Dockerfile using a `sed` command so the fix applies automatically when the container is built.

### 2. NumPy Version Incompatibility

PyTorch 2.1.2 was compiled against NumPy 1.x. When a newer NumPy 2.x was present in the environment, import errors occurred at runtime. This was resolved by pinning `numpy==1.26.4` in the requirements file.

### 3. Calling .eval() on the AEML Network Wrapper

The AEML `DukeMLP` class wraps a `Network` object, which in turn contains the actual PyTorch model at `model.model`. Calling `.eval()` directly on the wrapper raised an `AttributeError` because the wrapper class does not inherit from `nn.Module`. The fix was to extract the inner model explicitly with `net = model.model` and operate on `net` throughout training and evaluation.

### 4. LR Scheduler Stepping Behavior — The Main Training Issue

This was the most impactful issue. The original AEML training loop computes validation MSE every epoch and steps the `ReduceLROnPlateau` scheduler every epoch. When we rewrote `train.py` to add TensorBoard logging, we only computed validation every 10 epochs (the `EVAL_STEP` interval) and stepped the scheduler on that same cadence. This made the scheduler 10 times slower to react to plateaus, resulting in delayed learning rate reductions and significantly worse test MSE (0.003022 and 0.002410 in two runs) compared to Run 1 (0.001813). Additionally, in an intermediate version the scheduler was stepped on `train_mse` instead of `val_mse`, which is incorrect per the paper. The final fix was to compute validation MSE and step the scheduler every epoch, exactly matching the AEML loop, while still printing logs only every 10 epochs for readability.

### 5. snap-installed kubectl Cannot Find Authentication Plugins

On Ubuntu, `snap install kubectl` installs kubectl in a sandboxed environment with a restricted PATH. The `kubelogin` plugin, placed in `/usr/local/bin/`, was invisible to the snap version of kubectl, causing all `kubectl` commands to return silently with no output or error. The solution was to remove the snap version and install the official kubectl binary directly from the Kubernetes release page into `/usr/local/bin/`, which shares PATH with the plugin.

### 6. Docker Image Missing System Tools

The base image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` is a minimal runtime image and does not include `git`, `wget`, or `unzip`. The training entrypoint script required all three — `git` to clone the repository, `wget` to download the dataset, and `unzip` to extract it. These were added to the Dockerfile via `apt-get install`.

### 7. Python Output Buffering in Docker

When running the training container locally, no epoch logs appeared in the terminal even though the GPU was clearly active at 81% utilization. This was caused by Python's default output buffering, which holds stdout in memory until the buffer fills. The fix was to set the environment variable `PYTHONUNBUFFERED=1`, which forces Python to flush output immediately. This was added both to the `docker run` command and permanently to the Dockerfile via an `ENV` instruction.

### 8. Port Conflict for TensorBoard

Port 6006 was already in use by a local TensorBoard instance when attempting to expose TensorBoard from the Docker container. The container was mapped to port 6007 on the host instead (`-p 6007:6006`), and the same port-forwarding approach was used on Nautilus via `kubectl port-forward POD_NAME 6007:6006`.

### 9. kubectl cp on a Completed Pod

After training finished and the pod status changed to `Completed`, `kubectl cp` refused to copy files because it internally uses `kubectl exec`, which is not permitted on completed pods. The workaround was to spin up a temporary `busybox` pod with the results PVC mounted, keep it alive with a `sleep 300` command, and copy files from that pod instead.

### 10. Nautilus Resource Ratio Violation

The initial job YAML for the download job set memory requests to 4Gi and limits to 8Gi, and CPU requests to 2 with limits of 4. The Nautilus cluster policy requires that limits must not exceed requests by more than a factor of 1.2. A warning was raised and the fix was to set requests equal to limits for both jobs.

### 11. TensorBoard Port-Forward Failing on Nautilus

Attempting `kubectl port-forward` to access TensorBoard on the training pod failed because TensorBoard was not running inside the container — the training script only ran `train.py`. The solution was to exec into the running pod and start TensorBoard manually in a separate terminal, then port-forward from a second terminal.

### 12. Custom Training Loop Did Not Match the AEML Loop — Multiple Failed Runs

---

## Transformer Model — Architecture and Issues

### Transformer Architecture (Variant 1 — Paper Config)

The Transformer model uses the `DukeTransformer` class from AEML. The paper's exact configuration, recovered from the pre-trained `flags.obj` file, is:

- **Head MLP**: 8 layers projecting geometry input from 14 → 500 → ... → 6144 features, with BatchNorm + ReLU on all layers except the last
- **Reshape**: output reshaped to `[batch, 12, 512]` — 12 tokens of 512 dimensions (sequence_length × feature_channel_num)
- **Transformer Encoder**: 6 layers, 8 attention heads, feed-forward dimension 32
- **Tail linear**: single linear layer from 6144 → 2001 (no activation)
- **Training**: Adam lr=2e-4, weight_decay=5e-4, ReduceLROnPlateau factor=0.2, 300 epochs, batch_size=1024
- **Parameter count**: ~24.7 million
- **Paper's reported val MSE**: 0.001763 (from flags.obj `best_validation_loss`)
- **Paper's reported test MSE**: 0.001470 (Supplement Table 3)

### Transformer Issues Encountered

#### T1. save_model=False Default in AEML Transformer

Unlike the MLP wrapper, the AEML Transformer `Network.train_()` defaults to `save_model=False`. This meant the first two training runs completed without writing any checkpoint to disk. The fix is to always pass `save_model=True` explicitly. This is now set in `train_transformer.py`.

#### T2. NaN Loss at Epoch 50 — PyTorch Version Mismatch (Root Cause)

Every training attempt under `PyTorch 2.1.2` produced NaN loss values at or around epoch 50, regardless of the learning rate scheduler used. The loss would converge normally for the first 40–50 epochs and then collapse to NaN, making the model unrecoverable.

**Root cause**: The AEML Transformer code was written for PyTorch ~1.8/1.9 (the paper was published NeurIPS 2021). PyTorch 2.0 and 2.1 completely rewrote the `TransformerEncoder` internals, introducing Flash Attention, `scaled_dot_product_attention`, and nested tensor optimizations. These changes altered the numerical behavior of attention backward passes sufficiently to destabilize training for this architecture. The PyTorch warning seen during training — `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True` — is a direct symptom of this version mismatch.

**Attempts that did not fix this under PyTorch 2.x:**
- Switching from `warm_restart` to `reduce_plateau` scheduler
- Reducing learning rate from 2e-4 to 1e-4
- Adding gradient clipping (`clip_grad_norm_` with max_norm=1.0) to the AEML class_wrapper
- Increasing `ReduceLROnPlateau` patience from 10 to 30

**Fix**: Create a new conda environment (`adm_transformer`) and a new Docker image (`adm-train:transformer-v1`) using PyTorch 1.9.1+cu111. Training is stable and reaches epoch 300 without any NaN.

#### T3. distutils.version.LooseVersion Error with PyTorch 1.9.1

After downgrading to PyTorch 1.9.1, importing `from torch.utils.tensorboard import SummaryWriter` raised:

```
AttributeError: module 'distutils' has no attribute 'version'
```

PyTorch 1.9.1's bundled TensorBoard wrapper used `distutils.version.LooseVersion`, which was removed in `setuptools >= 60`. The fix is to pin `setuptools==59.5.0` before installing any other package. This is done in both the local conda env and the Dockerfile.

#### T4. CUDA Out of Memory with PyTorch 1.9.1 and batch_size=1024

Running with `batch_size=1024` on the local RTX 3070 (8 GB VRAM) under PyTorch 1.9.1 produced:

```
RuntimeError: CUDA out of memory. Tried to allocate 384.00 MiB
```

**Root cause**: The `DukeTransformer` model constructs input to the `TransformerEncoder` as `[batch, seq_len, embed]` = `[1024, 12, 512]`, but `TransformerEncoder` uses `batch_first=False` by default, so it interprets this as `seq_len=1024, batch=12`. The attention computation then operates on sequences of length 1024, producing attention weight tensors of shape `[8×12, 1024, 1024]` = 384 MB per layer — exactly matching the OOM error.

PyTorch 2.x's memory-efficient attention (Flash Attention) handled this silently without OOM, which is why the issue was invisible until switching to 1.9.1.

On the paper's cluster (Duke HPC, `/scratch/sr365/...`) with 32–80 GB GPU VRAM, this was never a problem at batch_size=1024.

**Fix**: For local runs on the 8 GB GPU, reduce batch_size to 256 via `--batch-size 256`. On Nautilus (A100/V100, 32+ GB VRAM), use the paper's batch_size=1024 via `--batch-size 1024`.

#### T5. warm_restart Scheduler Causing Loss Spikes

An early attempt used `lr_scheduler='warm_restart'` (CosineAnnealingWarmRestarts with T_0=50). This resets the learning rate to its initial value every 50 epochs, causing repeated loss spikes and preventing stable convergence. The paper's `flags.obj` confirms `lr_scheduler='reduce_plateau'` was used throughout. All variants now use `reduce_plateau`.

This was the most time-consuming issue across the Nautilus runs. After achieving Test MSE 0.001813 with the original AEML training loop locally (Run 1), we rewrote `train.py` with a custom loop to add detailed TensorBoard logging. This led to four consecutive runs on Nautilus with degraded results (Test MSE ranging from 0.002410 to 0.003209). The root cause was identified by reading the AEML `class_wrapper.py` source directly. The critical finding was on line 219: `self.lr_scheduler.step(train_avg_loss)` — the scheduler steps every epoch on training loss, not validation loss, and this line is outside the eval block. Our custom loop had been stepping on validation loss at different intervals across different attempts. Additionally, the AEML loop uses `reduction='mean'` averaged over batches rather than over total elements. After multiple failed attempts to replicate the behavior manually, the decision was made to abandon the custom loop entirely and call `ntwk.train_()` directly from the AEML `Network` class, with only a final test evaluation and `results.json` saving appended after. This is the safest approach as it uses the exact code the paper was trained with.

#### T6. Checkpoint Not Saved to PVC — Three Consecutive Failed Runs

Across the first three Nautilus Transformer runs, the trained checkpoint was never successfully saved to the `adm-results` PVC, despite training completing successfully each time.

**Run 1 (pod zxxvf):** Training crashed at the final test evaluation step with a CUDA OOM error (all 5,868 test samples fed at once — same batch_first=False issue as T4). Because the entrypoint script used `set -e`, the shell exited immediately on the Python error, skipping the `cp` command that followed. The PVC was left empty.

**Run 2 (pod wvjbh):** An in-pod file watcher was added to copy the checkpoint as soon as it appeared. The watcher failed silently — exact cause unclear, likely a race condition or path mismatch.

**Run 3 (pod s6d2z):** The OOM was fixed with batched evaluation (chunks of 256). The shell `cp` command ran and reported success, but the destination on the CephFS-backed PVC was created as an empty directory rather than a file. This is a known silent failure mode on CephFS when a copy appears to succeed but creates a directory node instead. The checkpoint was lost.

**Fix:** Move the checkpoint save entirely into Python using `shutil.copy2` at the end of `train_transformer.py`, before the script exits. Python's file I/O handles CephFS correctly and raises an exception if the copy fails, making failures visible rather than silent. The shell `cp` in `train_transformer.sh` was kept as a fallback but is no longer the primary save path. This fix was committed as `da2b7ee` and verified in the fourth run.

#### T7. OOM at Final Test Evaluation on All 5,868 Test Samples

After training completed, the final evaluation line `net(test_x_t)` passed all 5,868 test samples to the model in a single forward pass. Due to the batch_first=False bug (T4), the TransformerEncoder treated this as a sequence of length 5,868, producing attention weight tensors of shape `[8×12, 5868, 5868]` — several GB — causing OOM even on 32 GB VRAM.

**Fix:** Replaced the single-batch evaluation with a loop over chunks of 256 samples:
```python
_batch = 256
_preds = []
with torch.no_grad():
    for i in range(0, len(test_x_t), _batch):
        _preds.append(net(test_x_t[i:i+_batch].to(device)).cpu())
pred = torch.cat(_preds, dim=0)
```

#### T8. monitor_nautilus.sh Running as Multiple Instances

When the monitor script was restarted between runs without killing the previous instance, two processes polled simultaneously and wrote interleaved log lines, corrupting the log file. Fix: always run `pkill -f monitor_nautilus.sh` before restarting the monitor.

---

## Transformer Results

Four runs were conducted on Nautilus (namespace `gp-engine-malof`, A100/V100 GPU, batch_size=1024):

| Run | Pod | Outcome |
|-----|-----|---------|
| 1 | zxxvf | OOM at test eval, checkpoint lost |
| 2 | wvjbh | Watcher failed silently, checkpoint lost |
| 3 | s6d2z | CephFS silent failure, checkpoint lost |
| 4 | wmhwf | **Success** — checkpoint saved via Python shutil |

**Final results (Run 4):**

| Metric | Value |
|--------|-------|
| Best val MSE | 0.001502 (epoch 120) |
| Test MSE | **0.001501** |
| Paper target (Transformer) | 0.001470 |
| Gap | ~2% |

The model matches the paper's reported Transformer MSE within normal run-to-run variance. Training peaked at epoch 120, after which ReduceLROnPlateau reduced the LR at epoch 155 but validation loss did not improve further. The checkpoint is saved at `forward_model/models/Transformer/adm_transformer_v1_nautilus/best_model_forward.pt` and exported to ONNX at `forward_model/converted_models/transformer_v1.onnx`.
