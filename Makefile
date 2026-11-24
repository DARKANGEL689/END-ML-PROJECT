# Makefile for Substack Personalized Recommendation System Project

PYTHON = .venv/Scripts/python.exe
STREAMLIT = .venv/Scripts/streamlit.exe
PYTEST = .venv/Scripts/pytest.exe

.PHONY: setup data run test demo clean

setup:
	uv pip install -r requirements.txt --python $(PYTHON)

data:
	$(PYTHON) data/download_data.py

run:
	$(PYTHON) run_all.py

test:
	$(PYTEST) tests/ -v

demo:
	$(STREAMLIT) run app/streamlit_app.py

clean:
	rmdir /s /q results\models results\tables results\figures 2>nul || true
