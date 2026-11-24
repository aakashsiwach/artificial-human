from artificial_human.brain.evaluation import evaluate_text_similarity


def test_evaluation_scores_identical_reference_and_candidate_as_exact_match() -> None:
    result = evaluate_text_similarity(
        "Write a concise explanation with a practical example.",
        "Write a concise explanation with a practical example.",
    )
    assert result["exact_match"] is True
    assert result["token_f1"] == 1.0
    assert result["rouge_l_f1"] == 1.0
    assert result["metric_kind"] == "lexical_similarity"


def test_evaluation_reports_partial_overlap_and_token_truncation() -> None:
    result = evaluate_text_similarity(
        "A clear technical explanation with examples",
        "A concise technical explanation",
        max_tokens=4,
    )
    assert result["exact_match"] is False
    assert 0 < result["token_f1"] < 1
    assert result["truncated"] is True


def test_evaluation_handles_empty_token_sequences() -> None:
    result = evaluate_text_similarity("!!!", "???")
    assert result["exact_match"] is False
    assert result["token_f1"] == 0.0
    assert result["rouge_l_f1"] == 0.0
