from faq_rag.chunker import chunk_article
from faq_rag.parser import Article


def make_article(text: str) -> Article:
    return Article(
        url="https://help.dropbox.com/test/example",
        title="Example Title",
        category="Test",
        updated_date="Jan 01, 2025",
        text=text,
    )


def test_no_heading_becomes_a_single_overview_chunk():
    article = make_article("Just a short intro paragraph.")
    chunks = chunk_article(article)

    assert len(chunks) == 1
    assert chunks[0].section == "Overview"
    assert chunks[0].text.startswith("Example Title > Overview")
    assert "Just a short intro paragraph." in chunks[0].text


def test_splits_one_chunk_per_heading():
    text = "Intro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    chunks = chunk_article(make_article(text))

    assert [c.section for c in chunks] == ["Overview", "Section A", "Section B"]
    assert all(c.title == "Example Title" for c in chunks)
    assert all(c.category == "Test" for c in chunks)


def test_chunk_ids_are_unique_and_url_derived():
    text = "## A\n\nfoo\n\n## B\n\nbar"
    chunks = chunk_article(make_article(text))

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("test/example::") for cid in ids)


def test_long_section_splits_by_size_with_overlap():
    units = [f"This is paragraph number {i} with some filler words to pad it out nicely." for i in range(40)]
    text = "## Long Section\n\n" + "\n\n".join(units)
    chunks = chunk_article(make_article(text), max_chars=500, overlap_chars=100)

    assert len(chunks) > 1
    assert all(c.section == "Long Section" for c in chunks)
    # consecutive size-split pieces should share carried-over units (the overlap)
    for prev, curr in zip(chunks, chunks[1:]):
        prev_units = set(prev.text.split("\n\n"))
        curr_units = set(curr.text.split("\n\n"))
        assert prev_units & curr_units, "expected overlap between consecutive size-split chunks"


def test_size_split_pieces_are_made_of_whole_units_only():
    units = [f"Paragraph {i}: some filler words here to pad this out a bit." for i in range(30)]
    text = "## Big\n\n" + "\n\n".join(units)
    chunks = chunk_article(make_article(text), max_chars=300, overlap_chars=50)

    unit_set = set(units)
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]  # strip the "{title} > {section}" prefix line
        for piece in body.split("\n\n"):
            assert piece in unit_set, f"unexpected fragment (possible mid-unit cut): {piece!r}"
