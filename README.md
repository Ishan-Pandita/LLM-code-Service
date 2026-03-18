<div align="center">

# LLM Service

</div>

A production-ready **LLM inference service** built on [FastAPI](https://fastapi.tiangolo.com/) and [vLLM](https://github.com/vllm-project/vllm), designed for code analysis and automated code evaluation tasks. It exposes a modular REST API where each capability (best practices checking, iterative code fixing) is a self-contained module sharing one in-process model instance.

## Features

- **Single model, zero cold starts** — the model is loaded once on startup and shared across all concurrent requests via vLLM's continuous batching engine.
- **Module-based architecture** — each analysis capability is an isolated module with its own routes, schemas, and prompting logic. Adding a new module requires no changes to the core.
- **Function calling support** — modules prefer structured function-call output for reliable JSON extraction, with a fallback to regex-based JSON parsing.
- **Iterative code fixing** — the `evaluation_service` module integrates with a sandboxed [Judge0 compiler](./compiler/README.md) to verify each fix in a compiler-in-the-loop feedback cycle.
- **Deterministic by default** — all modules use greedy decoding (`temperature=0.0`) for reproducible, consistent results.
- **Full async pipeline** — built on `AsyncLLMEngine`, enabling parallel rule evaluations and concurrent request handling without blocking the event loop.

The **InferenceEngine** is a thread-safe singleton. It wraps the vLLM `AsyncLLMEngine` and exposes `generate()`, `generate_batch()`, and `generate_with_tools()`. All modules receive the same engine instance — there is only one model loaded per process.

The **ModuleRegistry** is a singleton that maps `module_id` strings to `BaseModule` subclasses. Modules self-register at import time.

## Modules

### `best_practices`

Evaluates code against a configurable set of coding best practice rules. Uses a **one-rule-per-generation** strategy: each rule is evaluated in a separate LLM call, and all calls are dispatched in parallel using `asyncio.gather`. This eliminates cross-rule contamination and JSON drift that occurs when evaluating multiple rules in a single prompt.

| Endpoint                                         | Method | Description                                            |
| ------------------------------------------------ | ------ | ------------------------------------------------------ |
| `/api/v1/modules/best_practices/rules`           | `GET`  | List all predefined rules (filterable by `?category=`) |
| `/api/v1/modules/best_practices/rules/{rule_id}` | `GET`  | Get a specific rule by ID                              |
| `/api/v1/modules/best_practices/categories`      | `GET`  | List all available rule categories                     |
| `/api/v1/modules/best_practices/evaluate`        | `POST` | Evaluate code against selected rules                   |
| `/api/v1/modules/best_practices/health`          | `GET`  | Module health check                                    |

**Evaluate request body:**

```json
{
  "language": "python",
  "code": "def foo(x):\n    return x*2",
  "predefined_rules": ["R1", "R3"],
  "custom_rules": [
    {
      "id": "C1",
      "name": "No magic numbers",
      "description": "All numeric literals should be assigned to named constants.",
      "category": "readability"
    }
  ]
}
```

**Response:**

```json
{
  "overall_status": "2/3",
  "rules": [
    {
      "rule_id": "R1",
      "status": "PASS",
      "confidence": "HIGH",
      "evidence": "Function name 'foo' is short but acceptable",
      "suggestion": null
    }
  ]
}
```

### `evaluation_service`

Accepts code that may contain errors and iteratively fixes it by running a **compiler-in-the-loop** feedback cycle:

1. Submit the code to the [Judge0 compiler](./compiler/README.md).
2. If the compiler reports an error, ask the LLM to fix **one** error at a time.
3. Re-submit the patched code and repeat until the compiler accepts it or `max_iterations` is reached.

This one-error-at-a-time approach produces more accurate fixes and generates a detailed fix history with severity scores and explanations — useful for educational tooling.

| Endpoint                                       | Method | Description                              |
| ---------------------------------------------- | ------ | ---------------------------------------- |
| `/api/v1/modules/evaluation_service/languages` | `GET`  | List languages supported by the compiler |
| `/api/v1/modules/evaluation_service/evaluate`  | `POST` | Evaluate and iteratively fix code        |
| `/api/v1/modules/evaluation_service/health`    | `GET`  | Compiler connectivity check              |

**Evaluate request body:**

```json
{
  "code": "def add(a, b)\n    return a + b",
  "language_id": 71,
  "language_name": "python",
  "max_iterations": 10,
  "stdin": null,
  "expected_output": "5\n",
  "expected_code": "def add(a, b):\n    return a + b"
}
```

**Response:**

```json
{
  "success": true,
  "total_errors_fixed": 1,
  "iterations_used": 2,
  "original_code": "def add(a, b)\n    return a + b",
  "fixed_code": "def add(a, b):\n    return a + b",
  "fixes": [
    {
      "iteration": 1,
      "error": {
        "error_type": "SyntaxError",
        "message": "invalid syntax",
        "line_number": 1
      },
      "original_code_snippet": "def add(a, b)",
      "fixed_code_snippet": "def add(a, b):",
      "explanation": "Missing colon at end of function definition.",
      "severity": 8
    }
  ],
  "final_output": "5\n"
}
```

## Getting Started

### Prerequisites

- Python 3.10+
- An NVIDIA GPU with CUDA support (required by vLLM)
- The [compiler service](./compiler/README.md) deployed and reachable (required by `evaluation_service` module)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ishan-Pandita/LLM-code-Service.git
cd LLM-code-Service

# Install the service and its dependencies
pip install -e .
```

For development tooling (linting, formatting, tests):

```bash
pip install -e ".[dev]"
```

### Configuration

Copy `.env.example` to `.env` and adjust the values for your environment:

```bash
cp .env.example .env
```

> [!IMPORTANT]
> Ensure the compiler service is deployed before starting the LLM Service if you intend to use the `evaluation_service` module. See the [Compiler Service README](./compiler/README.md) for instructions (`kubectl apply -f compiler/judge0-k8s.yaml`).

All settings use the `LLM_SERVICE_` prefix and can also be passed as environment variables. The most important ones:

| Variable                             | Default                  | Description                            |
| ------------------------------------ | ------------------------ | -------------------------------------- |
| `LLM_SERVICE_MODEL_ID`               | `zai-org/GLM-4.7-Flash`  | Hugging Face model identifier          |
| `LLM_SERVICE_TENSOR_PARALLEL_SIZE`   | `1`                      | Number of GPUs (set to your GPU count) |
| `LLM_SERVICE_GPU_MEMORY_UTILIZATION` | `0.90`                   | Fraction of GPU memory to use          |
| `LLM_SERVICE_MAX_MODEL_LEN`          | `32768`                  | Maximum context window length          |
| `LLM_SERVICE_QUANTIZATION`           | _(unset)_                | Optional: `awq`, `gptq`, `fp8`         |
| `LLM_SERVICE_COMPILER_BASE_URL`      | `http://localhost:32358` | URL of the Judge0 compiler service     |
| `LLM_SERVICE_API_PORT`               | `8000`                   | Port for the FastAPI server            |

### Running the service

```bash
# Via the installed script
llm-service

# Or directly with uvicorn
uvicorn llm_service.main:app --host 0.0.0.0 --port 8000
```

The service loads the model on startup. Depending on model size, this can take several minutes on first run (weights download from Hugging Face) or 30–120 seconds on subsequent runs. The API becomes available once the logs print `LLM Service Ready`.

Interactive API documentation is available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Docker

A `Dockerfile` is included for containerized deployment. It is based on the official `vllm/vllm-openai` image, which includes vLLM and its CUDA dependencies.

### Quick Start

1. **Clone and Configure**:

   ```bash
   git clone https://github.com/Ishan-Pandita/LLM-code-Service.git
   cd LLM-code-Service
   cp .env.example .env
   ```

2. **Build the Image**:

   ```bash
   docker build -t llm-service:latest .
   ```

3. **Run the Container**:
   ```bash
   docker run --gpus all \
     --env-file .env \
     -p 8000:8000 \
     llm-service:latest
   ```

> [!IMPORTANT]
> The container requires access to NVIDIA GPUs. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host.

For the `evaluation_service` module to reach the compiler, the container must be able to resolve `LLM_SERVICE_COMPILER_BASE_URL`. In a Docker Compose or Kubernetes setup, use the internal service DNS name instead of `localhost`.

## Adding a New Module

The module system is designed for extension. To add a new analysis capability:

1. Create a directory under `llm_service/modules/my_module/`.
2. Define `ModuleInput` and `ModuleOutput` Pydantic schemas in `schemas.py`.
3. Implement `BaseModule` in `module.py`, setting `module_id`, and implementing `build_system_prompt()`, `build_user_prompt()`, and `parse_output()`.
4. Call `get_module_registry().register(MyModule)` at the bottom of the module file.
5. Add module-specific FastAPI routes in `routes.py` and include the router in `llm_service/api/app.py`.

See `llm_service/modules/base.py` for the full interface contract and inline documentation.

## Stack

| Layer      | Technology                                                                            |
| ---------- | ------------------------------------------------------------------------------------- |
| Inference  | [vLLM](https://github.com/vllm-project/vllm) (AsyncLLMEngine)                         |
| Model      | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (default, configurable) |
| API        | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)        |
| Validation | [Pydantic v2](https://docs.pydantic.dev/latest/)                                      |
| Compiler   | [Judge0](https://judge0.com/) on Kubernetes                                           |
| Python     | 3.10+                                                                                 |

## Related

- [Compiler Service](./compiler/README.md) — Judge0 Kubernetes deployment required by the `evaluation_service` module.
