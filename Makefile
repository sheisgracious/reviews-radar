.PHONY: install run test clean

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && jupyter nbconvert --to notebook --execute reviews_radar_notebook.ipynb --output reviews_radar_notebook_executed.ipynb

test:
	. venv/bin/activate && python -m pytest tests/test_pipeline.py -v

clean:
	rm -f *.pkl model_metadata.json robinhood_reviews_app_store.csv
	rm -f *.png
	rm -f reviews_radar_notebook_executed.ipynb

