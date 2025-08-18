from augmentor.agents.validator import semantic_validate, deduplicate


def test_semantic_validate_empty():
    assert semantic_validate([]) == 1.0


def test_deduplicate_simple():
    texts = ["Hello", "Hello ", "World"]
    unique, dup_ratio = deduplicate(texts)
    assert len(unique) == 2
