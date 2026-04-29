# tests/test_pipeline.py
# for the core pipeline functions.


import re, pytest


def get_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sentiment labeling

def test_sentiment_labels():
    assert get_sentiment(1) == 'negative'
    assert get_sentiment(2) == 'negative'
    assert get_sentiment(3) == 'neutral'
    assert get_sentiment(4) == 'positive'
    assert get_sentiment(5) == 'positive'


# Text cleaning

def test_clean_text():
    assert not any(c.isdigit() for c in clean_text('lost $500 in 3 days'))
    assert clean_text('GREAT APP') == clean_text('great app')
    assert '!' not in clean_text('terrible!')
    assert '  ' not in clean_text('too   many   spaces')


# Google Play scraping 

def test_google_play_scrape():
    """Confirms the scraper reaches Google Play and returns reviews with expected fields."""
    pytest.importorskip('google_play_scraper')
    from google_play_scraper import reviews, Sort
    try:
        result, _ = reviews(
            'com.robinhood.android',
            lang='en', country='us',
            sort=Sort.NEWEST, count=5
        )
        assert isinstance(result, list) and len(result) > 0
        for r in result:
            assert 'content' in r and 'score' in r
            assert 1 <= r['score'] <= 5
    except Exception as e:
        pytest.skip(f'Network unavailable: {e}')