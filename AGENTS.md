# AGENTS.md

Project guidance for coding agents working in this repository.

## Runtime Environment

- On Windows, use the conda environment `droneRFa` for project commands.
- Before running Python scripts, tests, or dependency checks on Windows, activate it:
  ```powershell
  conda activate droneRFa
  ```
- On macOS, use the local virtual environment at `~/Desktop/venv`:
  ```bash
  ~/Desktop/venv/bin/python <script-or-command>
  ```
- Prefer commands that run from the repository root:
  - Windows: `E:\code\python\droneRFa`
  - macOS: `~/Desktop/code/uavIdentification`

## Working Style

- Think before coding. State assumptions when a request is ambiguous.
- Ask before implementing if multiple interpretations could change the result.
- Prefer the simplest change that fully satisfies the request.
- Do not add speculative features, abstractions, configuration, or broad error handling.

## Editing Rules

- Make surgical changes. Touch only files and lines needed for the task.
- Match the existing style and structure of the project.
- Add necessary comments for non-obvious code, especially complex logic, important assumptions, GPU/parallel computation details, or behavior that is easy to misuse.
- Keep comments concise and useful. Do not add comments that merely repeat what the code already says.
- Do not refactor, reformat, or clean up unrelated code.
- Remove only unused imports, variables, or files introduced by your own change.
- If unrelated dead code or suspicious behavior is found, mention it instead of editing it.

## Verification

- Define success criteria before non-trivial changes.
- For bug fixes, reproduce the issue when practical, then verify the fix.
- For new behavior, add or run focused tests when the repository supports it.
- Run verification inside the `droneRFa` conda environment.
- On macOS, run verification with `~/Desktop/venv`.
- If verification cannot be run, explain exactly why.

## Planning

For multi-step work, use a brief plan:

1. Describe the change and its success check.
2. Implement the smallest necessary edit.
3. Run the relevant verification.

These rules are intended to keep diffs small, behavior clear, and project state easy to review.
