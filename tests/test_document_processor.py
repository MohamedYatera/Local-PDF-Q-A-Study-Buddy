from app.document_processor import infer_section_title, split_into_chunks


def test_split_into_chunks_uses_overlap():
    words = " ".join(f"word{i}" for i in range(20))
    chunks = split_into_chunks(words, chunk_size_words=8, overlap_words=2)

    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == ["word6", "word7"]
    assert chunks[1].split()[:2] == ["word6", "word7"]


def test_infer_section_title_prefers_short_heading():
    text = "Dynamic Programming\nThis lecture introduces memoization."
    assert infer_section_title(text, 5) == "Dynamic Programming"


def test_infer_section_title_falls_back_to_page():
    text = "This line is too long to be treated as a section heading because it exceeds the heading length limit."
    assert infer_section_title(text, 2) == "Page 2"

