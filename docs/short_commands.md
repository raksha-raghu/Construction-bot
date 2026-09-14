# Common commands

Run these from the project root after activating the virtual environment.

```powershell
python -m pip install -r requirements/pinned.txt
python -m backend --rebuild
python -m backend "Your question"
python -m backend.evaluation
python -m backend.evaluation --method bm25
python -m unittest discover -s tests -v
```
