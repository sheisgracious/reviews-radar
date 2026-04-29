.PHONY: install run test clean

install:
	pip install app-store-web-scraper google-play-scraper langdetect contractions scikit-learn pandas matplotlib seaborn nbconvert jupyter pytest 

run:
	jupyter nbconvert --to notebook --execute reviews_radar_notebook.ipynb --output reviews_radar_notebook_executed.ipynb --ExecutePreprocessor.timeout=600

test:
	python -m pytest tests/test_pipeline.py -v

clean:
	rm -f *.pkl model_metadata.json robinhood_reviews_app_store.csv *.png
	rm -f reviews_radar_notebook_executed.ipynb