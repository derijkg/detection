import os

IGNORE = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "data",
    "checkpoints",
    "models",
    ".ipynb_checkpoints",
    "wandb",
    "evals"
}


def print_tree(dir_path=".", prefix=""):
    items = sorted(os.listdir(dir_path))
    items = [i for i in items if i not in IGNORE]
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = i == len(items) - 1
        print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
        if os.path.isdir(path):
            print_tree(
                path, prefix + ("    " if is_last else "│   ")
            )  # limits deep recursion naturally


print_tree()