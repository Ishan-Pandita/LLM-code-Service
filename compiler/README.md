# Compiler Service

A production-grade, sandboxed **code execution backend** built on [Judge0](https://judge0.com/), deployed to Kubernetes. It handles secure, isolated code submission and execution for the LLM Service's `evaluation_service` module — providing the compiler-in-the-loop feedback that drives iterative code fixing.

> [!NOTE]
> This component is a dependency of the parent [LLM Service](../README.md). The `evaluation_service` module talks to this service at the URL configured via `LLM_SERVICE_COMPILER_BASE_URL`.

## Overview

The compiler runs as a set of Kubernetes workloads (Namespace, Deployments, Services) all defined in a single manifest file. The stack consists of:

| Component | Image | Role |
|---|---|---|
| `judge0-server` | `judge0/judge0:latest` | REST API for code submission and result polling |
| `judge0-workers` | `judge0/judge0:latest` | Background workers that execute submitted code in isolation |
| `postgres` | `postgres:13` | Persistent store for submissions and results |
| `redis` | `redis:7` | Queue broker between the server and workers |

The server is exposed on **NodePort 32358**, which is the default `LLM_SERVICE_COMPILER_BASE_URL` (`http://localhost:32358`) configured in the parent service.

## Prerequisites

- A running Kubernetes cluster (local: [minikube](https://minikube.sigs.k8s.io/), [kind](https://kind.sigs.k8s.io/), [k3d](https://k3d.io/); cloud: GKE, EKS, AKS)
- `kubectl` configured and pointed at your cluster

## Deployment

### 1. Generate a secret key

The Judge0 server requires a `SECRET_KEY_BASE`. Generate a secure random value and set it in the manifest before deploying:

```bash
# Generate a secure random key (requires openssl)
openssl rand -hex 64
```

Open `judge0-k8s.yaml` and paste the value into the `SECRET_KEY_BASE` field of the `judge0-secret`:

```yaml
stringData:
  SECRET_KEY_BASE: "<your-generated-key-here>"
```

### 2. Apply the manifest

```bash
kubectl apply -f judge0-k8s.yaml
```

This creates the `judge0` namespace and all resources within it. You can verify everything is running:

```bash
kubectl get all -n judge0
```

### 3. Wait for readiness

The `judge0-server` has a readiness probe on `GET /languages`. Wait until all pods report `Running` and the server pod is `Ready`:

```bash
kubectl rollout status deployment/judge0-server -n judge0
```

> [!IMPORTANT]
> The `judge0-server` pod takes approximately **30-60 seconds** to initialize (database migrations run at startup). The `judge0-workers` pod requires privileged mode (`securityContext.privileged: true`) to run sandboxed processes — ensure your cluster permits this.

## Verifying the Service

Once deployed, the server is reachable on the node's IP at port `32358`. From within the cluster or via a local tunnel:

```bash
# List all supported language IDs (sanity check)
curl http://localhost:32358/languages | python -m json.tool
```

You should receive a JSON array of language objects like `[{"id": 71, "name": "Python (3.8.1)"}, ...]`.

To submit a test code execution:

```bash
curl -X POST http://localhost:32358/submissions \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "print(\"Hello, World!\")",
    "language_id": 71,
    "wait": true
  }'
```

A successful response will include `"status": {"id": 3, "description": "Accepted"}` and `"stdout": "Hello, World!\n"`.

## Architecture Notes

**One-error-at-a-time loop:** The parent service's `evaluation_service` module does not rely on the compiler for batch processing. Instead, it submits code, reads a single error from the response, asks the LLM to fix it, and re-submits. This loop repeats up to `max_iterations` times or until the compiler returns `Accepted`.

**No persistent volumes:** The Postgres and Redis deployments use `emptyDir` volumes. All submission history is lost on pod restart. This is intentional for a stateless evaluation service — only the final fixed code matters to the caller.

**Worker privileges:** The `judge0-workers` pod runs with `privileged: true` because the Judge0 worker needs to create isolated Linux namespaces (via `isolate`) for each code execution. Restrict cluster access accordingly.

## Resource Allocation

| Deployment | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---|---|---|---|---|
| `postgres` | 250m | 500m | 256Mi | 512Mi |
| `redis` | 100m | 250m | 128Mi | 256Mi |
| `judge0-server` | 250m | 1 | 512Mi | 1Gi |
| `judge0-workers` | 500m | 2 | 512Mi | 1Gi |

Adjust limits in `judge0-k8s.yaml` based on your expected concurrency and available cluster resources.

## Teardown

```bash
kubectl delete namespace judge0
```

This removes all resources created by the manifest.

