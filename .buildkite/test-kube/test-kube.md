Intended Goal
Primary Objective: Convert the existing VM-based Buildkite CI setup to use the Buildkite Agent Stack for Kubernetes (agent-stack-k8s) on GKE as a Proof of Concept (PoC).
Target Hardware: Run tests on available Google Cloud TPU v6e (Trillium) capacity using the single-chip ct6e-standard-1t node type (1x1 topology).
Target Workload: Execute a representative end-to-end small model inference test (examples/offline_inference.py with Qwen/Qwen3-0.6B) using a pre-built Docker CI container image (vllm-tpu).
Summary of Issues Encountered, Attempted Solutions, and Step Examples
1. ModuleNotFoundError: No module named 'vllm'
Issue: Running python3 examples/offline_inference.py failed to import vllm. The Docker image installed vllm using editable mode (pip install -e /workspace/vllm). In agent-stack-k8s, Kubernetes mounts an emptyDir volume at /workspace for every container in the Pod, which masked the Docker image's internal /workspace/vllm folder.
Tried Solutions:
checkout.dir & custom volumeMounts: Attempted setting checkout.dir: /workspace/tpu_inference or mountPath: /workspace/tpu_inference in podSpec. The agent-stack-k8s controller hardcodes mountPath: /workspace for the workspace volume at the controller level, ignoring step-level mount overrides.
Script execution in /tmp: Cloned vllm source to /tmp/vllm and ran pip install --no-deps -e /tmp/vllm to re-link Python's .pth file to the pre-compiled C++ binaries in site-packages.
Buildkite Step Example (Issue 1):
yaml
steps:
  - label: ":kubernetes: E2E Offline Inference (Qwen3-0.6B)"
    key: "kube_e2e_qwen3_0_6b"
    agents:
      queue: "kube"
    env:
      PYTHONUNBUFFERED: "1"
    commands:
      - echo "=== Starting E2E Offline Inference PoC on agent-stack-k8s ==="
      - |
        python3 examples/offline_inference.py \
          --model Qwen/Qwen3-0.6B \
          --tensor-parallel-size 1 \
          --max-model-len 1024 \
          --max-tokens 128
    plugins:
      - kubernetes:
          podSpec:
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: "tpu-v6e-slice"
              cloud.google.com/gke-tpu-topology: "1x1"
            tolerations:
              - key: "google.com/tpu"
                operator: "Equal"
                value: "present"
                effect: "NoSchedule"
            containers:
              - name: vllm-tpu-runner
                image: "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/vllm-tpu:0cbe535e57dbd8da6b83561fe44a4e9018bd5d7d-bd091079cba0800d8b8ee8ed22feab5864d1b101-tpu6e"
                resources:
                  limits:
                    google.com/tpu: "1"
                  requests:
                    google.com/tpu: "1"
                volumeMounts:
                  - name: dshm
                    mountPath: /dev/shm
            volumes:
              - name: dshm
                emptyDir:
                  medium: Memory
                  sizeLimit: 16Gi
2. No such file or directory & checkout.skip: true Behavior
Issue A: Setting workingDir: /workspace/tpu_inference in podSpec was overridden by buildkite-agent bootstrap script, which hardcodes cd /workspace/build at startup.
Issue B: Enabling checkout.skip: true disabled git checkout. Because /workspace was masked by the empty K8s volume and git checkout was skipped, /workspace/build remained empty, resulting in python3: can't open file ... [Errno 2] No such file or directory.
Diagnostic Check: A filesystem search (find / -name "offline_inference.py") confirmed /workspace/build was empty when checkout.skip: true was enabled.
Buildkite Step Example (Issue 2):
yaml
steps:
  - label: ":kubernetes: E2E Offline Inference (Qwen3-0.6B)"
    key: "kube_e2e_qwen3_0_6b"
    agents:
      queue: "kube"
    env:
      PYTHONUNBUFFERED: "1"
    commands:
      - echo "=== Starting E2E Offline Inference PoC on agent-stack-k8s ==="
      - |
        python3 examples/offline_inference.py \
          --model Qwen/Qwen3-0.6B \
          --tensor-parallel-size 1 \
          --max-model-len 1024 \
          --max-tokens 128
    plugins:
      - kubernetes:
          checkout:
            skip: true
          podSpec:
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: "tpu-v6e-slice"
              cloud.google.com/gke-tpu-topology: "1x1"
            tolerations:
              - key: "google.com/tpu"
                operator: "Equal"
                value: "present"
                effect: "NoSchedule"
            containers:
              - name: vllm-tpu-runner
                image: "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/vllm-tpu:0cbe535e57dbd8da6b83561fe44a4e9018bd5d7d-bd091079cba0800d8b8ee8ed22feab5864d1b101-tpu6e"
                resources:
                  limits:
                    google.com/tpu: "1"
                  requests:
                    google.com/tpu: "1"
                volumeMounts:
                  - name: dshm
                    mountPath: /dev/shm
            volumes:
              - name: dshm
                emptyDir:
                  medium: Memory
                  sizeLimit: 16Gi

I want the commands section to keep clean, as an entry point that people just write cmd and run their tests and just work. How shoud I adjust the pipeline? We can consider updating Dockerfile later but not now