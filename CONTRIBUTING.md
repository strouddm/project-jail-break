# Contributing

Hello, trying to keep it simple here and learn some best practices along the way!

## Getting started and making changes

1. Clone the repository (see [README.md](README.md) for the full setup and data-download steps).

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create a branch for your work:

   ```bash
   git checkout -b your-feature-name
   ```

4. Make your changes (see [Where things go](#where-things-go) and [Conventions](#conventions) below), then run any relevant scripts/notebooks to confirm your change works. Stage and commit your work:

   ```bash
   git add <files>
   git commit -m "Add WildGuardMix loader"
   ```

5. Make sure your branch is up to date with `main`, then push your branch:

   ```bash
   git fetch origin
   git rebase origin/main
   git push -u origin your-feature-name
   ```

   If you rebase again after this first push, the branch has diverged from the remote, so use `git push --force-with-lease`.

6. Open a pull request with a short description of what changed and why:

   ```bash
   gh pr create --fill
   ```

7. Wait for **at least one approving review** before merging — anyone can review. Address any comments by pushing follow-up commits, then merge once you have that approval and all checks pass:

   ```bash
   git add <files>   # only needed if you added new files
   git commit -m "Address review comments"
   git push
   gh pr merge --squash
   ```

## Where things go

| Directory     | What belongs here                                       |
| ------------- | ------------------------------------------------------- |
| `src/`        | Reusable Python modules (data prep, features, models, eval) |
| `scripts/`    | Runnable CLIs (download, train, evaluate)               |
| `notebooks/`  | EDA and experiments                                     |
| `reports/`    | Plots and result artifacts                              |

Do **not** commit data or model artifacts — `data/raw/`, `data/processed/`, and `*.pkl`/`*.pt`/`*.bin` files are gitignored on purpose.

## Conventions

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Use clear, descriptive names.
- **Notebooks:** Clear outputs before committing to keep diffs small.
- **Commits:** Write concise, present-tense messages (e.g. "Add WildGuardMix loader").
- **Imports:** Keep `src/` modules importable and free of side effects at import time.
- Delete branch when complete

