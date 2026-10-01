import json

from faq_rag.indexer import load_articles


def test_load_articles_keeps_the_newest_record_per_url(tmp_path):
    path = tmp_path / "articles.jsonl"
    base = {"title": "T", "category": "C", "updated_date": None}
    records = [
        {**base, "url": "https://help.dropbox.com/x/a", "text": "old parse"},
        {**base, "url": "https://help.dropbox.com/x/b", "text": "only version"},
        {**base, "url": "https://help.dropbox.com/x/a", "text": "new parse", "parser_version": 2},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))

    articles = {a.url: a for a in load_articles(path)}
    assert len(articles) == 2
    assert articles["https://help.dropbox.com/x/a"].text == "new parse"
    assert articles["https://help.dropbox.com/x/a"].parser_version == 2
    assert articles["https://help.dropbox.com/x/b"].parser_version == 1
