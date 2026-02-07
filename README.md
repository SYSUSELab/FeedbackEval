# FeedbackEval: Evaluating Large Language Models in Feedback-Driven Code Repair

This is the official repository for the paper "FeedbackEval: Evaluating Large Language Models in Feedback-Driven Code Repair".

## Repository Structure

The repository is organized as follows:

```
Feedback/
├── SWE-bench/                # (Manual Download Required) Official SWE-bench framework
├── dataset/                  # Dataset files (CoderEval, HumanEval, SWE-Bench-verified)
├── input/                    # Input configurations and raw data
├── src/
│   ├── code/                 # Main source code
│   │   ├── core/             # Core utilities, client wrapper, and patch generation
│   │   ├── evaluation/       # Evaluator strategies (Local & SWE-bench)
│   │   ├── feedback/         # Feedback generation and handling logic
│   │   ├── mutations/        # Mutant generation for test feedback
│   │   ├── prompts/          # Prompt template builders
│   │   ├── retrieval/        # Retrieval utilities (BM25)
│   │   ├── swebench_ext/     # SWE-bench extension and helpers
│   │   └── evaluate.py       # Main entry point for evaluation
│   ├── scripts/              # Shell scripts for running experiments
│   └── templates/            # Jinja2 prompt templates
├── results/                  # Generated repairs and evaluation results
├── logs/                     # Execution logs
└── requirements.txt          # Python dependencies
```

## Benchmark Overview

We construct a new benchmark, **FeedbackEval**, to systematically evaluate LLMs’ ability to interpret and utilize various feedback types in code repair.

**Datasets Supported:**
- **HumanEval**
- **CoderEval**
- **SWE-Bench-verified** (Requires Docker)

**Feedback Types:**
* **Compiler Feedback**: Syntax errors and code style violations.
* **Test Feedback**: Failing tests and expected outcomes.
* **Minimal Feedback**: Concise failure notification (e.g., "The code is wrong").
* **LLM-Skilled Feedback**: Natural language suggestions from a "skilled" persona.
* **LLM-Expert Feedback**: Precise, targeted suggestions from an "expert" persona.
* **Mixed Feedback**: A composite of multiple feedback sources.

## Setup

### 1. Prerequisites
- **Python 3.10+**
- **Docker**: Required for `SWE-Bench-verified` and `CoderEval`.

### 2. Docker Environments

**SWE-Bench-verified:**
The system automatically handles Docker containers. Just ensure the Docker Daemon is running.

**CoderEval:**
For the `CoderEval` dataset, the evaluation code uses a specific runtime environment. You must setup the official CoderEval Docker container:
1. Obtain the Docker environment from [CoderEval GitHub](https://github.com/CoderEval/CoderEval).
2. Copy this project into the container directory `/home/travis/builds`:
   ```bash
   docker cp <path_to_FeedbackEval> <container_id>:/home/travis/builds
   ```
3. Run the evaluation scripts inside this container.

### 3. Configuration
Create a `.env` file in the project root to configure environment variables (e.g., API keys, concurrency settings).
```bash
# .env example
OPENAI_API_KEY=sk-...
MAX_WORKERS=4
```

### 3. Repository Setup

**Download SWE-bench Framework:**
Because this project relies on the official SWE-bench evaluation harness, you must manually clone or download the official [SWE-bench repository](https://github.com/swe-bench/SWE-bench) and place it in the root of this project.

Structure should look like:
```
Feedback/
  ├── SWE-bench/
  │    ├── swebench/
  │    └── ... 
  └── src/
```

**Install Dependencies:**
Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

*Note: For `SWE-Bench-verified`, the system will automatically handle Docker container creation and management.*

## Usage

We provide shell scripts in `src/scripts/` to streamline the evaluation process. Ensure you are in the project root or adjust paths accordingly.

### 1. Single-Round Repair
Perform a single round of code repair and evaluation.

```bash
# Run repair generation and execution
bash src/scripts/single_fix.sh

# Calculate pass rates
bash src/scripts/single_score.sh
```

### 2. Multi-Round Repair
Perform iterative repair with feedback loops (up to 3 rounds by default).

```bash
# Run iterative repair
bash src/scripts/multi_fix.sh

# Calculate pass rates across rounds
bash src/scripts/multi_score.sh
```

### 3. Configuration
You can customize the execution by modifying the scripts or setting environment variables:

- **`MAX_WORKERS`**: Controls concurrency for SWE-bench Docker evaluations (default: 4).
- **`MODELS`**: Map of model names to versions in the scripts.
- **`DATASET`**: Select target datasets (`SWE-Bench-verified`, `HumanEval`, `CoderEval`).

Example modification in `src/scripts/single_fix.sh`:
```bash
DATASET=("SWE-Bench-verified")
FEEDBACK_TYPES=("test_feedback")
```