from faq_rag.parser import PARSER_VERSION, parse_article

ARTICLE_HTML = """
<html><body>
<div class="aem-Grid aem-Grid--12">
  <ul class="hp03-breadcrumb__list">
    <li class="hp03-breadcrumb__item">Dropbox Help Center</li>
    <li class="hp03-breadcrumb__item">Test Category</li>
    <li class="hp03-breadcrumb__item">Test Article Title</li>
  </ul>
  <div class="dmep-plank-frame hp01-header-updated-date aem-GridColumn">
    <h1 class="hp01-header-updated-date__headline">Test Article Title</h1>
    <div class="hp01-header-updated-date__updated-date">Updated Jan 01, 2025</div>
  </div>
  <div class="responsivegrid aem-GridColumn">
    <div>
      <p>Intro with <strong>bold text</strong> and a <a href="https://example.com">link</a>.</p>
      <h2>First Section</h2>
      <p>Some content.</p>
      <ul><li>Bullet one</li><li>Bullet two</li></ul>
      <h3>Sub Section</h3>
      <ol><li>Step one</li><li>Step two</li></ol>
      <svg><title>decorative icon</title><path d="M0 0"/></svg>
    </div>
  </div>
  <div class="dwg-plank-frame hp04-helpfulness aem-GridColumn">Was this article helpful?</div>
  <div class="dwg-plank-frame hp13-related-articles aem-GridColumn">Related Articles</div>
  <div class="dwg-plank-frame hp14-community-answers aem-GridColumn">Community answers</div>
</div>
<div class="responsivegrid">Unrelated footer content that should NOT be picked up</div>
</body></html>
"""


def test_extracts_title_category_and_date():
    article = parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example")
    assert article is not None
    assert article.title == "Test Article Title"
    assert article.category == "Test Category"
    assert article.updated_date == "Jan 01, 2025"
    assert article.url == "https://help.dropbox.com/test/example"


def test_renders_headings_lists_and_bold():
    article = parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example")
    assert "## First Section" in article.text
    assert "### Sub Section" in article.text
    assert "- Bullet one" in article.text
    assert "- Bullet two" in article.text
    assert "1. Step one" in article.text
    assert "2. Step two" in article.text
    assert "**bold text**" in article.text


def test_bold_keeps_the_space_that_sits_inside_the_tag():
    html = ARTICLE_HTML.replace(
        "<p>Some content.</p>",
        "<p>Click <strong>Admin console </strong>in the sidebar.</p>",
    )
    article = parse_article(html, "https://help.dropbox.com/test/example")
    assert "Click **Admin console** in the sidebar." in article.text


def test_drops_link_hrefs_but_keeps_link_text():
    article = parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example")
    assert "link" in article.text
    assert "example.com" not in article.text


def test_excludes_svg_helpfulness_related_and_community_sections():
    article = parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example")
    assert "decorative icon" not in article.text
    assert "Was this article helpful" not in article.text
    assert "Related Articles" not in article.text
    assert "Community answers" not in article.text


def test_ignores_unrelated_responsivegrid_outside_the_articles_grid():
    article = parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example")
    assert "Unrelated footer content" not in article.text


def _page(body_html: str) -> str:
    """Minimal page with the real article scaffold around `body_html`."""
    return f"""<html><body><div class="aem-Grid">
      <div class="hp01-header-updated-date"><h1 class="hp01-header-updated-date__headline">T</h1></div>
      <div class="responsivegrid">{body_html}</div>
    </div></body></html>"""


def test_table_rows_become_self_contained_bullets():
    html = _page(
        "<table><tr><th>User issue</th><th>Possible solution</th></tr>"
        "<tr><td>I forgot my password</td><td>Reset <strong>your password</strong>. See "
        "<a href='https://x'>How to reset</a>.</td></tr></table>"
    )
    text = parse_article(html, "https://help.dropbox.com/t/t").text
    assert text == (
        "- User issue: I forgot my password; Possible solution: Reset **your password**. See How to reset."
    )


def test_table_without_header_joins_cells():
    html = _page("<table><tr><td>Basic</td><td>3 devices</td></tr><tr><td>Plus</td><td>unlimited</td></tr></table>")
    text = parse_article(html, "https://help.dropbox.com/t/t").text
    assert text == "- Basic | 3 devices\n\n- Plus | unlimited"


def test_keeps_text_that_is_not_inside_a_paragraph():
    html = _page("<div>Choose an option:<a href='https://x'>Reset password</a></div><div><p>After</p></div>")
    text = parse_article(html, "https://help.dropbox.com/t/t").text
    assert text == "Choose an option:Reset password\n\nAfter"


def test_drops_html_comments():
    html = _page("<p>Visible<!-- internal note --> text</p>")
    assert parse_article(html, "https://help.dropbox.com/t/t").text == "Visible text"


def test_articles_record_the_parser_version():
    assert parse_article(ARTICLE_HTML, "https://help.dropbox.com/test/example").parser_version == PARSER_VERSION


def test_returns_none_when_h1_is_missing():
    assert parse_article("<html><body><p>Not an article</p></body></html>", "https://help.dropbox.com/x") is None
