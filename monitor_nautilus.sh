#!/bin/bash
# Monitors the Nautilus Transformer training job, copies results when done,
# and deletes the job to free cluster resources.

NAMESPACE="gp-engine-malof"
JOB_NAME="adm-train-transformer"
LOCAL_DEST="/home/qubit/malof_lab/Project_1/forward_model/models/Transformer/adm_transformer_v1_nautilus"
LOG="/home/qubit/malof_lab/Project_1/monitor_nautilus.log"
POLL_INTERVAL=120  # seconds

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== Monitor started. Job: $JOB_NAME, Namespace: $NAMESPACE ==="

while true; do
    STATUS=$(kubectl get pod -n "$NAMESPACE" \
        -l job-name="$JOB_NAME" \
        --no-headers 2>/dev/null | awk '{print $3}' | head -1)

    log "Pod status: ${STATUS:-unknown}"

    if [[ "$STATUS" == "Completed" || "$STATUS" == "Error" ]]; then
        log "Pod finished with status: $STATUS. Saving logs and copying results..."

        # Save full pod logs
        POD_NAME=$(kubectl get pod -n "$NAMESPACE" -l job-name="$JOB_NAME" --no-headers | awk '{print $1}' | head -1)
        kubectl logs "$POD_NAME" -n "$NAMESPACE" \
            > "/home/qubit/malof_lab/Project_1/forward_model/results/nautilus_transformer_run.log" 2>>"$LOG" \
            && log "Pod logs saved to forward_model/results/nautilus_transformer_run.log" \
            || log "WARNING: could not save pod logs"

        # Spin up a busybox pod to access the results PVC
        kubectl run nautilus-copy-helper \
            --image=busybox \
            --restart=Never \
            --overrides='{
              "spec": {
                "automountServiceAccountToken": false,
                "volumes": [{
                  "name": "adm-results",
                  "persistentVolumeClaim": {"claimName": "adm-results"}
                }],
                "containers": [{
                  "name": "busybox",
                  "image": "busybox",
                  "command": ["sleep", "300"],
                  "volumeMounts": [{
                    "name": "adm-results",
                    "mountPath": "/results"
                  }]
                }]
              }
            }' \
            -n "$NAMESPACE" 2>>"$LOG"

        log "Waiting for busybox pod to be ready..."
        kubectl wait pod/nautilus-copy-helper \
            --for=condition=Ready \
            --timeout=120s \
            -n "$NAMESPACE" 2>>"$LOG"

        # Copy checkpoint from PVC to local
        mkdir -p "$LOCAL_DEST"
        kubectl cp \
            "$NAMESPACE/nautilus-copy-helper:/results/models/adm_transformer_v1/best_model_forward.pt" \
            "$LOCAL_DEST/best_model_forward.pt" 2>>"$LOG" \
            && log "Checkpoint copied to $LOCAL_DEST" \
            || log "ERROR: checkpoint copy failed — check PVC manually"

        # Delete busybox helper pod
        kubectl delete pod nautilus-copy-helper -n "$NAMESPACE" 2>>"$LOG"
        log "Busybox helper deleted."

        # Delete the training job to free cluster resources
        kubectl delete job "$JOB_NAME" -n "$NAMESPACE" 2>>"$LOG"
        log "Training job $JOB_NAME deleted from cluster."

        log "=== All done. Check $LOCAL_DEST for the checkpoint. ==="
        break
    fi

    sleep "$POLL_INTERVAL"
done
