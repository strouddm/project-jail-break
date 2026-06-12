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

Never commit directly to main. Every feature or fix gets its own branch, gets reviewed via a Pull Request, then merges into main.

* Before making any changes, sync your local main with the remote (NEVER skip this or you may run into merge conflicts later on)

```bash
    git checkout main
    git fetch origin
    git reset --hard origin/main
```

* When you are ready to make a change to the code, create new branch (this code switches you to the newly created branch automatically)

```bash
    git checkout -b <branch name>
```

* Do your work and commit

```bash
    git add <file>
    git commit -m "description"
```

* Push the branch to GitHub

```bash
    git push -u origin <branch name>
```

* Visit GitHub and open a pull request

    Pull requests allow the team to review your changes before integrating them into the main branch. In the comment box you can tag reviewers and leave a description of the changes you made. GitHub will show reviewers every line of code you changed. After discussing with the team, make any requested changes, then push again to update the PR.

* Approve the pull request

    Someone on the team will have to be the one to officially accept the changes. Communicate with the team to decide who this will be.

    To learn more about pull requests, visit [this link.](https://github.blog/developer-skills/github/beginners-guide-to-github-creating-a-pull-request/)

* After the PR is approved and merged, clean up locally

```bash
    git checkout main
    git pull
    git branch -d <branch name>
```

The final three commands switch back to main, pull down the newly merged changes, then `git branch -d` deletes the feature branch since its work is now in main. The `-d` flag stands for delete — it won't delete unless the branch has already been merged, so it's safe to run.

### Key Points

- Multiple people can work simultaneously on separate branches without interfering with each other.
- Pull Requests are for discussion, not just merging — open one early if you need feedback before you're done.
- `main` should always be stable and deployable.
- Start a new branch for each new piece of work, repeating from Step 1.

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

