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

   By default the harness calls DeepSeek's API directly (recommended:
   cheapest, official, no re-hosting layer). Edit `.env` and set
   `DEEPSEEK_API_KEY` to a key from https://platform.deepseek.com/api_keys

   To route through OpenRouter instead (useful if you want automatic
   fallback across hosts, or plan to compare DeepSeek against other models
   later), set `DEEPSEEK_PROVIDER=openrouter` and `OPENROUTER_API_KEY` to a
   key from https://openrouter.ai/keys

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

## Codebase translation agent

`harness/translate_agent.py` translates an entire codebase from one
language to another, file by file, keeping cross-file imports consistent:

```bash
python3 cli.py translate \
  --src ./myapp --out ./myapp-js \
  --from python --to javascript \
  --verify --run "node main.js"
```

- `--src` / `--out`: source codebase and where to write the translation.
- `--from` / `--to`: any of `python, javascript, typescript, go, ruby,
  java, rust, php, c, cpp, csharp`.
- Files are discovered recursively (skipping `node_modules`, `.git`,
  `venv`, `dist`, build output, etc.), translated in a dependency-friendly
  order (leaf files first, `main`/`index`/`app`-style entry points last),
  and non-source files (docs, configs, assets) are copied over as-is
  unless `--no-copy-other` is passed.
- Each file's prompt includes a running manifest of already-translated
  sibling paths, so later files import the new filenames/extensions
  instead of the original ones.
- `--verify --run "<cmd>"` (optionally with `--install "<cmd>"` and
  `--network` if that install needs internet) executes the translated
  project inside the same kind of disposable, resource-capped Docker
  sandbox used for scoring tasks above -- this is the actual proof the
  translation runs, not just that it looks plausible.

This is a best-effort translation, not a compiler: review the output for
anything language-specific the model may have approximated (concurrency
primitives, standard-library quirks, package-manager manifests), and use
`--verify` with your project's real test command whenever one exists.

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
cli.py                  entrypoint
harness/
  client.py              DeepSeek API wrapper (task solving + file translation)
  sandbox.py             Docker sandbox runner (single-file tasks + whole projects)
  runner.py              orchestrates client + sandbox, records results
  translate_agent.py     whole-codebase translation agent
tasks/                  task definitions (prompt + tests)
results/                JSON result records (gitignored)
```
