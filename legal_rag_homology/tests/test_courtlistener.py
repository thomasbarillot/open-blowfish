from unittest.mock import patch, MagicMock
from scripts.legal_rag_homology.corpus import courtlistener


def _mock_response(status, json_body=None, text_body=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body or {}
    r.text = text_body
    r.raise_for_status = MagicMock()
    return r


def test_lookup_citation_returns_opinion_id_on_hit():
    with patch.object(courtlistener, "_get") as mget:
        mget.return_value = _mock_response(200, {
            "count": 1,
            "results": [{"citation": "384 U.S. 436", "clusters": [{"id": 111, "sub_opinions": [{"id": 222}]}]}],
        })
        result = courtlistener.lookup_citation("384 U.S. 436")
        assert result == 222


def test_lookup_citation_returns_none_on_miss():
    with patch.object(courtlistener, "_get") as mget:
        mget.return_value = _mock_response(200, {"count": 0, "results": []})
        assert courtlistener.lookup_citation("999 U.S. 999") is None


def test_fetch_opinion_returns_plain_text():
    with patch.object(courtlistener, "_get") as mget:
        mget.return_value = _mock_response(200, {
            "id": 222, "plain_text": "Opinion body.",
            "cluster": {"id": 111, "docket": {"court_id": "scotus"},
                        "date_filed": "1966-06-13",
                        "citations": [{"volume": 384, "reporter": "U.S.", "page": 436}]},
        })
        op = courtlistener.fetch_opinion(222)
        assert op["opinion_id"] == "222"
        assert op["text"] == "Opinion body."
        assert op["court"] == "scotus"
        assert op["date_filed"] == "1966-06-13"
        assert "384 U.S. 436" in op["all_citations"]


def test_get_retries_on_429_then_raises_court_listener_error():
    from scripts.legal_rag_homology.corpus.courtlistener import CourtListenerError
    with patch.object(courtlistener.requests, "get") as mget, \
         patch.object(courtlistener.time, "sleep") as msleep:
        mget.return_value = _mock_response(429)
        try:
            courtlistener._get("http://example/x")
            raised = False
        except CourtListenerError:
            raised = True
        assert raised
        assert mget.call_count == 3
        assert msleep.call_count == 3


def test_fetch_bulk_court_opinions_paginates_via_next():
    page1 = _mock_response(200, {
        "next": "http://example/next",
        "results": [
            {"id": 1, "plain_text": "opinion one",
             "cluster": {"date_filed": "2000-01-01", "docket": {"court_id": "scotus"}}},
            {"id": 2, "plain_text": "", "cluster": {}},  # skipped (no text)
        ],
    })
    page2 = _mock_response(200, {
        "next": None,
        "results": [
            {"id": 3, "plain_text": "opinion three",
             "cluster": {"date_filed": "2001-01-01", "docket": {"court_id": "ca9"}}},
        ],
    })
    with patch.object(courtlistener, "_get") as mget:
        mget.side_effect = [page1, page2]
        out = courtlistener.fetch_bulk_court_opinions("scotus", max_records=10)
        assert [o["opinion_id"] for o in out] == ["1", "3"]
        assert mget.call_count == 2
        first_call_kwargs = mget.call_args_list[0].kwargs
        second_call_kwargs = mget.call_args_list[1].kwargs
        assert first_call_kwargs.get("params") == {"cluster__docket__court": "scotus", "page_size": 100}
        assert second_call_kwargs.get("params") is None
