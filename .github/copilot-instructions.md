# Copilot Instructions for this Repo

- Prefer PyTorch unless asked otherwise.
- Always support CUDA when available (torch.cuda.is_available()).
- Keep code modular: src/data.py, src/models.py, src/train.py, src/eval.py, src/utils.py.
- Avoid global state; pass config objects explicitly.
- Use pathlib for paths; never hardcode absolute paths.
- Add type hints for public functions and dataclasses for configs.
- When adding dependencies, update requirements.txt and explain why.
- Provide minimal, testable functions; add pytest tests for utilities.
