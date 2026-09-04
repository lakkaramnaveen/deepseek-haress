# deepseek-harness

A small test harness for evaluating DeepSeek's coding models: it sends coding
tasks to DeepSeek's official API, then executes the model's generated code
inside an isolated, disposable Docker container (a "complete OS" sandbox) to
check whether it actually passes the task's tests.

Uses DeepSeek's official API at `api.deepseek.com` (OpenAI-compatible).
Not affiliated with any third-party "DeepSeek Harness" site — those are not
official DeepSeek properties.

## Setup

1. **Install dependencies**

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Install Docker Desktop** (required for the sandbox that runs generated
   code): https://www.docker.com/products/docker-desktop/
   Start Docker Desktop before running tasks.

3. **Add your API key**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `DEEPSEEK_API_KEY` to a key from
   https://platform.deepseek.com/api_keys

## Usage

```bash
# List available tasks
python3 cli.py list

# Run every task
python3 cli.py run

# Run a single task by id
python3 cli.py run --task fizzbuzz
```

Each run prints a PASS/FAIL summary and writes a full JSON record (prompt,
generated code, stdout/stderr, timing) to `results/`.

## How it works

1. `harness/client.py` sends the task prompt to DeepSeek's chat completions
   API and extracts the returned code block.
2. `harness/sandbox.py` writes that code plus the task's test script into a
   throwaway directory, then runs it inside a fresh Docker container with
   `--network none` and capped CPU/memory, so generated code can't touch
   your host, the network, or leftover state from a previous run.
3. `harness/runner.py` orchestrates the above per task and records the
   result.

## Adding a task

Add a new YAML file under `tasks/`:

```yaml
id: my_task
language: python   # or javascript
prompt: |
  Write a Python function `foo(x)` that ...
test_code: |
  from solution import foo
  assert foo(1) == 2
  print("HARNESS_PASS")
```

`test_code` runs inside the sandbox with the model's code available as
`solution` (Python: `from solution import ...`, JS: `require("./solution.js")`).
A non-zero exit code (e.g. a failed `assert`) counts as a failed task.

## Project layout

```
cli.py              entrypoint
harness/
  client.py          DeepSeek API wrapper
  sandbox.py          Docker sandbox runner
  runner.py           orchestrates client + sandbox, records results
tasks/               task definitions (prompt + tests)
results/             JSON result records (gitignored)
```
