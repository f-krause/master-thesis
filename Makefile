# Linting
#format:
#	@echo "🖨️ Format code: Running ruff"
#	@uvx ruff format

# type checking
#mypy:
#	@uv run mypy "$(CURDIR)/src"

# packing
pack:
	@echo "🗂️ Packaging code into flatfile - use as knowledge base for Claude/aider/etc."
	@uvx repopack "$(CURDIR)/src" --ignore *lock*,.github/*,.mypy_cache/*,architecture-diagram*,*.svg,*.ipynb,pretraining/motif_db/* --output "codebase.txt"