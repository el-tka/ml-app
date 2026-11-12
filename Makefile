.PHONY: run install clean check runner
.DEFAULT_GOAL: = runner

run: install
	cd src && poetry run python runner.py

install: pyproject.toml
	poetry install

clean:
	poetry run python -c "import pathlib, shutil, os, stat; fix = lambda f: (os.chmod(f, stat.S_IWRITE | stat.S_IREAD), os.unlink(f)); [(shutil.rmtree(p, onerror=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE | stat.S_IREAD), func(path)))) for p in pathlib.Path('.').rglob('__pycache__') if p.exists()]"
	
check: 
	poetry run flake8 src/

runner: check run clean
