import pandas as pd
from scripts.legal_rag_homology import dataset


def test_extract_questions_returns_100_unique_rows(monkeypatch):
    fake_rows = [
        {"Question ID": f"q{i // 4}", "Question Category": "Rule QA",
         "Question": f"question text {i // 4}", "Model": "Westlaw",
         "Response": "r", "Correctness": "Correct", "Groundedness": "Grounded",
         "Label": "Accurate"}
        for i in range(400)
    ]
    fake_df = pd.DataFrame(fake_rows)
    monkeypatch.setattr(dataset, "_load_hf_dataset", lambda: fake_df)

    qs = dataset.extract_questions()
    assert len(qs) == 100
    assert set(qs.columns) == {"question_id", "category", "question_text"}
    assert qs["question_id"].is_unique


def test_extract_released_responses_preserves_all_rows(monkeypatch):
    fake_rows = [
        {"Question ID": f"q{i // 4}", "Question Category": "Rule QA",
         "Question": f"qt {i // 4}", "Model": "Westlaw",
         "Response": f"r{i}", "Correctness": "Correct",
         "Groundedness": "Grounded", "Label": "Accurate"}
        for i in range(400)
    ]
    fake_df = pd.DataFrame(fake_rows)
    monkeypatch.setattr(dataset, "_load_hf_dataset", lambda: fake_df)

    rr = dataset.extract_released_responses()
    assert len(rr) == 400
    assert "tool" in rr.columns
    assert "human_label" in rr.columns
