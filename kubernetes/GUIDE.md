# Nautilus Cluster Setup Guide — ADM Forward Model

## Overview

```
Local machine                     Nautilus (gp-engine-malof)
─────────────────                 ──────────────────────────
Docker image ──push──► DockerHub ◄──pull── Training Job
                                            │
                               ┌────────────┼────────────┐
                               │            │            │
                          adm-data    adm-results   /develop/code
                          (PVC 5Gi)   (PVC 5Gi)   (cloned in job)
```

**Directory layout inside every container:**

| Path | Contents |
|---|---|
| `/develop/data/ADM/` | ADM dataset CSVs (from adm-data PVC) |
| `/develop/results/` | Model checkpoints and plots (adm-results PVC) |
| `/develop/code/` | Repo cloned fresh at job start |

---

## Files in this directory

| File | Purpose |
|---|---|
| `pvc.yaml` | Creates adm-data and adm-results PVCs |
| `Dockerfile` | MLP training image — PyTorch 2.1.2+cu121 |
| `Dockerfile.transformer` | Transformer training image — PyTorch 1.9.1+cu111 |
| `job-download-data.yaml` | One-time job: downloads + extracts ADM dataset to PVC |
| `job-train.yaml` | MLP training job |
| `job-train-transformer.yaml` | Transformer training job (Variant 1, batch_size=1024) |
| `scripts/download_data.sh` | Runs inside the download job |
| `scripts/train.sh` | Runs inside the MLP training job |
| `scripts/train_transformer.sh` | Runs inside the Transformer training job |

> **Why two Dockerfiles?** The AEML Transformer code was written for PyTorch ~1.9. Running it under PyTorch 2.x causes NaN loss at epoch ~50 due to changes in the TransformerEncoder backend. The MLP is unaffected by this and continues to use PyTorch 2.1.2. See `docs/issues_and_model.md` (section T2) for the full diagnosis.

---

## Step 1 — Nautilus account and kubectl

### 1.1 Create account and get namespace access
- Go to **https://nrp.ai/** and log in with your university account
- Accept the Acceptable Use Policy on the main portal page
- Email **Alex Hurt (jhurt@missouri.edu)** to be added to `gp-engine-malof`

### 1.2 Install kubectl (official binary — NOT snap)
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client
```

> **Note:** Do not use `snap install kubectl` — the snap sandbox prevents it from finding authentication plugins.

### 1.3 Install kubelogin plugin (required for authentication)
```bash
curl -LO https://github.com/int128/kubelogin/releases/latest/download/kubelogin_linux_amd64.zip
unzip kubelogin_linux_amd64.zip
sudo mv kubelogin /usr/local/bin/kubectl-oidc_login
rm kubelogin_linux_amd64.zip LICENSE README.md
kubectl oidc-login --help   # should print help text
```

### 1.4 Download kubeconfig and authenticate
```bash
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL "https://nrp.ai/config"
kubectl config use-context nautilus
kubectl config set contexts.nautilus.namespace gp-engine-malof

# This opens a browser — log in with your university account
kubectl get pods -n gp-engine-malof
# Expected: "No resources found in gp-engine-malof namespace."
```

---

## Step 2 — Create PVCs

```bash
cd /path/to/Project_1
kubectl create -f kubernetes/pvc.yaml

# Verify both show STATUS=Bound
kubectl get pvc -n gp-engine-malof
```

Expected output:
```
NAME          STATUS   CAPACITY   STORAGECLASS
adm-data      Bound    5Gi        rook-cephfs-central
adm-results   Bound    5Gi        rook-cephfs-central
```

> PVCs may show Pending initially — they become Bound once a pod uses them.

---

## Step 3 — Build Docker image

Base image: `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` (DockerHub, matches exact PyTorch version).
The Dockerfile installs git, wget, unzip, AEML, and patches the `verbose=True` bug in AEML's ReduceLROnPlateau.

### 3.1 Build locally
```bash
cd /path/to/Project_1/kubernetes

docker build --no-cache --platform linux/x86_64 \
    -f Dockerfile \
    -t mahadkhaliq/adm-train:v1 .
```

### 3.2 Test locally before pushing
```bash
docker run --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -p 6007:6006 \
  -v /path/to/Project_1/data:/develop/data \
  -v /path/to/Project_1/forward_model/results:/develop/results \
  mahadkhaliq/adm-train:v1 \
  bash -c "bash /develop/train.sh & \
           tensorboard --logdir /develop/code/forward_model/models/MLP/adm_mlp --host 0.0.0.0 --port 6006 & \
           wait"
```

Open `http://localhost:6007` for TensorBoard. Kill with Ctrl+C after a few epochs.

### 3.3 Push to DockerHub
```bash
docker login
docker push mahadkhaliq/adm-train:v1
```

### 3.4 Clean up local Docker images
```bash
docker stop $(docker ps -q)
docker rm $(docker ps -aq)
docker rmi IMAGE_ID   # remove images no longer needed
```

---

## Step 4 — Download ADM dataset to PVC (one-time only)

Data source: `https://research.repository.duke.edu/record/176/files/ADM.zip?ln=en`

```bash
kubectl create -f kubernetes/job-download-data.yaml -n gp-engine-malof

# Watch pod status
kubectl get pods -n gp-engine-malof

# Stream logs (replace POD_NAME with actual name from above)
kubectl logs -f POD_NAME -n gp-engine-malof
```

When complete, the PVC will contain:
```
/develop/data/
└── ADM/
    ├── data_g.csv
    ├── data_s.csv
    └── testset/
        ├── test_g.csv
        └── test_s.csv
```

Delete the job when done:
```bash
kubectl delete job adm-download-data -n gp-engine-malof
```

---

## Step 5 — Run training

```bash
kubectl create -f kubernetes/job-train.yaml -n gp-engine-malof

# Watch pod status
kubectl get pods -n gp-engine-malof

# Stream training logs
kubectl logs -f POD_NAME -n gp-engine-malof
```

### Monitor TensorBoard live
```bash
kubectl port-forward POD_NAME 6007:6006 -n gp-engine-malof
```
Then open `http://localhost:6007`.

Training takes ~2 hours. Results are saved to the `adm-results` PVC automatically.

> If the pod stays **Pending**, resources are unavailable. Check available GPUs at
> **https://portal.nrp-nautilus.io/resources** and reduce your request if needed.
> Keep GPU utilization >40% or you may get flagged.

Delete the job when done:
```bash
kubectl delete job adm-train -n gp-engine-malof
```

---

## Step 6 — Download results

The pod must be **Running** (not Terminated) to use `kubectl cp`.

```bash
# Copy model checkpoint
kubectl cp gp-engine-malof/POD_NAME:/develop/results/models/adm_mlp \
    ./forward_model/models/MLP/adm_mlp_nautilus

# Copy training plots
kubectl cp gp-engine-malof/POD_NAME:/develop/results/train_results \
    ./forward_model/results/nautilus/
```

---

## Helpful commands

| Task | Command |
|---|---|
| List pods | `kubectl get pods -n gp-engine-malof` |
| Stream logs | `kubectl logs -f POD_NAME -n gp-engine-malof` |
| Pod details (Pending/Error) | `kubectl describe pod POD_NAME -n gp-engine-malof` |
| Shell into running pod | `kubectl exec -it POD_NAME -n gp-engine-malof -- /bin/bash` |
| Delete a job | `kubectl delete job JOB_NAME -n gp-engine-malof` |
| List PVCs | `kubectl get pvc -n gp-engine-malof` |
| TensorBoard port-forward | `kubectl port-forward POD_NAME 6007:6006 -n gp-engine-malof` |

> **Important:** Never delete jobs or pods that don't belong to you.

---

## Resource limits (must satisfy limits ≤ requests × 1.2)

| Job | CPU | Memory | GPU |
|---|---|---|---|
| adm-download-data | 4 | 8Gi | none |
| adm-train | 4 | 16Gi | 1 |

---

## GPU utilization policy

| Resource | Rule |
|---|---|
| GPU | Must stay **>40%** while job is running |
| CPU | Must stay between **20%–200%** |
| Memory | Must stay between **20%–150%** |

Delete idle jobs immediately — do not leave jobs running with 0% GPU usage.

---

---

## Running the Transformer on Nautilus

### Build and push the Transformer image

```bash
cd /path/to/Project_1/kubernetes

docker build --no-cache --platform linux/x86_64 \
    -f Dockerfile.transformer \
    -t mahadkhaliq/adm-train:transformer-v1 .

docker push mahadkhaliq/adm-train:transformer-v1
```

> The Transformer image uses PyTorch 1.9.1+cu111 and setuptools==59.5.0.
> Do not use the MLP image (`adm-train:v1`) for Transformer training.

### Submit the training job

The data PVC (`adm-data`) already contains the ADM dataset from the MLP run — no re-download needed.

```bash
kubectl create -f kubernetes/job-train-transformer.yaml -n gp-engine-malof

# Watch pod status
kubectl get pods -n gp-engine-malof

# Stream training logs
kubectl logs -f POD_NAME -n gp-engine-malof
```

Training runs for 300 epochs at batch_size=1024. Expected val MSE target: ~0.001763 (paper's reported value from flags.obj).

### Download results

```bash
kubectl cp gp-engine-malof/POD_NAME:/develop/results/models/adm_transformer_v1 \
    ./forward_model/models/Transformer/adm_transformer_v1_nautilus
```

Delete the job when done:
```bash
kubectl delete job adm-train-transformer -n gp-engine-malof
```

### Running locally (reduced batch size)

On an 8 GB GPU, use batch_size=256 to avoid OOM:

```bash
conda activate adm_transformer  # PyTorch 1.9.1 env
python forward_model/train_transformer.py --variant 1 --batch-size 256
```

---

## Acknowledgment (for papers)

> "Computational resources for this research have been supported by the NSF National Research Platform, as part of GP-ENGINE (award OAC #2322218)."
