.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute reviews_radar_notebook.ipynb --output executed_notebook.ipynb

test:
	python -m pytest tests/test_pipeline.py -v

clean:
	rm -f models/*.pkl model_metadata.json robinhood_reviews_app_store.csv
	rm -f figures/*.png
	rm -f reviews_radar_notebook_executed.ipynb