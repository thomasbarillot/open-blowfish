import json
from scripts.legal_rag_homology.evaluation import citation_parser


def _load(fixtures_dir, key):
    data = json.loads((fixtures_dir / "sample_answers.json").read_text())
    return data[key]


def test_extract_citations_real_single(fixtures_dir):
    case = _load(fixtures_dir, "real_single")
    cits = citation_parser.extract_citations(case["text"])
    assert [c.normalized for c in cits] == case["expected_citations"]


def test_extract_citations_two_citations(fixtures_dir):
    case = _load(fixtures_dir, "two_citations")
    cits = citation_parser.extract_citations(case["text"])
    assert [c.normalized for c in cits] == case["expected_citations"]


def test_extract_citations_no_citation(fixtures_dir):
    case = _load(fixtures_dir, "no_citation")
    cits = citation_parser.extract_citations(case["text"])
    assert cits == []


def test_extract_citations_captures_adjacent_quote(fixtures_dir):
    case = _load(fixtures_dir, "real_with_quote")
    cits = citation_parser.extract_citations(case["text"])
    assert len(cits) == 1
    assert "right of privacy" in (cits[0].adjacent_quote or "")


def test_extract_citations_accepts_unicode_smart_quotes():
    text = (
        "As stated in Roe v. Wade, 410 U.S. 113 (1973), "
        "“the right of privacy is broad.”"
    )
    cits = citation_parser.extract_citations(text)
    assert len(cits) == 1
    assert "right of privacy" in (cits[0].adjacent_quote or "")


def test_extract_citations_skips_partial_citations():
    text = "Some reference to U.S. 436 without a volume, and Miranda v. Arizona, 384 U.S. 436 (1966)."
    cits = citation_parser.extract_citations(text)
    normalized = [c.normalized for c in cits]
    assert "384 U.S. 436" in normalized
    assert all(c.volume and c.page and c.reporter for c in cits)


def test_extract_citations_associates_each_citation_with_its_own_quote(fixtures_dir):
    case = _load(fixtures_dir, "two_citations_with_quotes")
    cits = citation_parser.extract_citations(case["text"])
    normalized = [c.normalized for c in cits]
    assert normalized == case["expected_citations"]
    by_norm = {c.normalized: c for c in cits}
    assert "privacy is broad" in (by_norm["410 U.S. 113"].adjacent_quote or "")
    assert "commerce is plenary" in (by_norm["22 U.S. 1"].adjacent_quote or "")
