# tests/test_sources_ranking.py
from magicat.modules.sources.providers import ImageMatch
from magicat.modules.sources.ranking import domain_of, rank_matches


def m(url, score=0.5, provider="p"):
    return ImageMatch(source_url=url, score=score, provider=provider)


def test_domain_of():
    assert domain_of("https://www.tiktok.com/@u/video/1") == "tiktok.com"
    assert domain_of("https://m.youtube.com/watch?v=1") == "youtube.com"
    assert domain_of("https://blog.example.co/post") == "blog.example.co"
    assert domain_of("not a url") == ""


def test_rank_orders_by_score_desc():
    ranked = rank_matches([m("https://a/1", 0.2), m("https://b/2", 0.9)])
    assert [r.source_url for r in ranked] == ["https://b/2", "https://a/1"]


def test_rank_dedupes_by_domain_keeping_best():
    ranked = rank_matches([
        m("https://www.tiktok.com/@u/video/1", 0.9),
        m("https://www.tiktok.com/@u/video/2", 0.7),   # same domain: dropped
        m("https://youtube.com/watch?v=1", 0.6),
    ])
    assert [r.source_url for r in ranked] == [
        "https://www.tiktok.com/@u/video/1",
        "https://youtube.com/watch?v=1",
    ]


def test_rank_caps_results():
    matches = [m(f"https://site{i}.example/x", 1.0 - i * 0.1)
               for i in range(8)]
    assert len(rank_matches(matches, limit=5)) == 5


def test_rank_empty():
    assert rank_matches([]) == []
