
project_name = reddit_analyser


venv: .venv/.touchfile

.venv/.touchfile: requirements.txt
	python -m venv .venv
	source .venv/bin/activate; python -m pip install --upgrade pip; pip install -Ur requirements.txt
	touch .venv/.touchfile

requirements.txt:
	touch requirements.txt

run: venv
	source .venv/bin/activate; python $(project_name)/$(project_name).py

freeze: venv
	source .venv/bin/activate; pip freeze > requirements.txt

list: venv
	source .venv/bin/activate; pip list

clean:
	rm -rf .venv