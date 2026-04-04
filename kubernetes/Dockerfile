FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /develop

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y git wget unzip && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# torch is already in the base image — we only add what's missing.
# After installing AEML==0.0.1 we patch out verbose=True from
# ReduceLROnPlateau, which was removed in PyTorch 2.x.
RUN pip install --no-cache-dir \
        AEML==0.0.1 \
        "numpy<2" \
        pandas \
        scikit-learn \
        matplotlib \
        tensorboard \
        tqdm \
        einops \
        seaborn && \
    AEML_PATH=$(python -c "import AEML, os; print(os.path.dirname(AEML.__file__))") && \
    sed -i 's/verbose=True,\s*//g' "$AEML_PATH/models/MLP/class_wrapper.py"

# Copy entrypoint scripts into the image
COPY scripts/download_data.sh /develop/download_data.sh
COPY scripts/train.sh /develop/train.sh
RUN chmod +x /develop/download_data.sh /develop/train.sh
