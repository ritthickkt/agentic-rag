"""The agent loop. Scripted LLM, so these assert control flow and cost no money.

The properties tested here are the ones that cost real money or real trust when they
break: how many LLM calls a path makes, whether evidence accumulates, and whether
"I could not find it" is ever reported when the system in fact produced an answer.
"""

from __future__ import annotations

import pytest

from agentic_rag.agent import Answerer
from agentic_rag.models import Chunk, RetrievedChunk
from tests.conftest import ScriptedLLM


class StubRetriever:
    """Returns a scripted list of chunks per hop and records every query."""

    def __init__(self, per_hop: list[list[str]]) -> None:
        self.per_hop = per_hop
        self.queries: list[str] = []

    def search(self, query, k=None, doc_ids=None, hop=0):
        self.queries.append(query)
        texts = self.per_hop[min(hop, len(self.per_hop) - 1)]
        return [
            RetrievedChunk(
                chunk=Chunk(chunk_id=f"h{hop}-{i}", doc_id="d", text=t, ordinal=i,
                            page=i + 1, filename="doc.pdf"),
                score=1.0 - i / 100,
                dense_rank=i + 1,
                hop=hop,
            )
            for i, t in enumerate(texts)
        ]


def answered(answer, citations, confidence="high"):
    return {"insufficient": False, "answer": answer, "citations": citations,
            "missing": "", "confidence": confidence}


def insufficient(missing):
    return {"insufficient": True, "answer": "", "citations": [], "missing": missing,
            "confidence": "low"}


class TestSingleHop:
    def test_sufficient_evidence_answers_in_one_call(self, cfg):
        retr = StubRetriever([["Revenue was $47.2 million in Q3."]])
        llm = ScriptedLLM([answered("Revenue was $47.2 million [1].",
                                    [{"passage": 1, "quote": "Revenue was $47.2 million"}])])
        res = Answerer(retriever=retr, llm=llm).answer("What was revenue?")

        assert res.abstained is False
        assert res.n_llm_calls == 1          # no grader call, no wasted hop
        assert len(res.hops) == 1
        assert res.citations[0].verified is True
        assert res.confidence == "high"

    def test_empty_index_abstains_with_no_llm_call(self, cfg):
        """Nothing to read means nothing to pay for."""
        retr = StubRetriever([[]])
        llm = ScriptedLLM([])               # any call would raise
        res = Answerer(retriever=retr, llm=llm).answer("anything?")

        assert res.abstained is True
        assert res.n_llm_calls == 0
        assert "index may be empty" in res.abstain_reason


class TestMultiHop:
    def test_reformulates_toward_the_missing_fact(self, cfg):
        retr = StubRetriever([
            ["The film Alien was released in 1979."],
            ["Ridley Scott is a British director."],
        ])
        llm = ScriptedLLM([
            insufficient("the name of the director of Alien"),
            {"query": "Alien 1979 directed by"},
            answered("Ridley Scott, who is British [2].",
                     [{"passage": 2, "quote": "Ridley Scott is a British director"}]),
        ])
        res = Answerer(retriever=retr, llm=llm).answer("What nationality is Alien's director?")

        assert res.abstained is False
        assert len(res.hops) == 2
        assert res.n_llm_calls == 3          # answer + reformulate + answer
        # The second query must target the gap, not repeat the question.
        assert retr.queries[1] == "Alien 1979 directed by"

    def test_evidence_accumulates_across_hops(self, cfg):
        """Hop two must add to hop one, never replace it."""
        retr = StubRetriever([
            ["First hop passage about the film."],
            ["Second hop passage about the director."],
        ])
        llm = ScriptedLLM([
            insufficient("the director"),
            {"query": "director name"},
            answered("Answer [1].", [{"passage": 1, "quote": "First hop passage"}]),
        ])
        res = Answerer(retriever=retr, llm=llm).answer("q?")

        texts = " ".join(rc.chunk.text for rc in res.chunks)
        assert "First hop" in texts and "Second hop" in texts

    def test_repeated_query_stops_the_loop_instead_of_burning_a_call(self, cfg):
        """Re-running a query returns the same chunks; another hop learns nothing.

        The reformulator here hands back the original question verbatim, which is the
        realistic failure: it has nothing new to work with, so it restates the ask.
        """
        retr = StubRetriever([["Unrelated passage."]])
        llm = ScriptedLLM([
            insufficient("the dividend policy"),
            {"query": "What is the dividend policy?"},   # == the original question
        ])
        res = Answerer(retriever=retr, llm=llm, max_hops=3).answer(
            "What is the dividend policy?"
        )

        assert res.abstained is True
        assert res.n_llm_calls == 2           # answer + reformulate, then stop
        assert len(res.hops) == 1             # never searched a second time
        assert res.abstain_reason == "the dividend policy"

    def test_grader_with_no_missing_fact_stops(self, cfg):
        """'Insufficient' with nothing named gives the next hop nothing to search."""
        retr = StubRetriever([["Something."]])
        llm = ScriptedLLM([insufficient("")])
        res = Answerer(retriever=retr, llm=llm, max_hops=3).answer("q?")

        assert res.abstained is True and res.n_llm_calls == 1


class TestExhaustedLoop:
    def test_final_hop_is_told_it_is_final(self, cfg):
        """The measured bug this guards: abstaining because the budget ran out,
        while holding the evidence. The last attempt must know it is the last."""
        retr = StubRetriever([["A"], ["B"], ["C"]])
        llm = ScriptedLLM([
            insufficient("x"), {"query": "q2"},
            insufficient("y"), {"query": "q3"},
            answered("Partial answer [1].", [{"passage": 1, "quote": "A"}]),
        ])
        res = Answerer(retriever=retr, llm=llm, max_hops=3).answer("q?")

        assert len(res.hops) == 3
        assert res.abstained is False
        # Only the final answer prompt carries the note.
        notes = [c for c in llm.calls if "final set of passages" in c["prompt"]]
        assert len(notes) == 1

    def test_partial_answer_is_kept_not_discarded(self, cfg):
        """A model that flags insufficient but writes an answer has still answered."""
        retr = StubRetriever([["Revenue was $47.2 million in Q3 2025."]])
        llm = ScriptedLLM([{
            "insufficient": True,
            "answer": "Q3 2025 revenue was $47.2 million [1]; 2024 is not covered.",
            "citations": [{"passage": 1, "quote": "Revenue was $47.2 million"}],
            "missing": "the 2024 figure",
            "confidence": "medium",
        }])
        res = Answerer(retriever=retr, llm=llm, max_hops=1).answer("Revenue in 2024 and 2025?")

        assert res.abstained is False
        assert "47.2" in res.answer
        assert res.confidence == "low"           # downgraded, but delivered
        assert res.abstain_reason == "the 2024 figure"


class TestCitationHandling:
    def test_unverifiable_citation_is_flagged_not_hidden(self, cfg):
        retr = StubRetriever([["The CTO is Priya Raman."]])
        llm = ScriptedLLM([answered(
            "She came from Northwind [1].",
            [{"passage": 1, "quote": "She previously worked at Northwind Systems"}],
        )])
        res = Answerer(retriever=retr, llm=llm).answer("Where did the CTO work before?")

        assert res.citations[0].verified is False
        assert res.verification["n_unverified"] == 1
        assert "[1]" in res.answer          # the marker stays; the chip shows a warning

    def test_marker_with_no_citation_is_stripped_from_the_prose(self, cfg):
        """A marker pointing at nothing reads as evidence the answer does not have."""
        retr = StubRetriever([["Revenue was $47.2 million."]])
        llm = ScriptedLLM([answered(
            "Revenue was $47.2 million [1], and margin improved [5].",
            [{"passage": 1, "quote": "Revenue was $47.2 million"}],
        )])
        res = Answerer(retriever=retr, llm=llm).answer("q?")

        assert "[1]" in res.answer
        assert "[5]" not in res.answer

    def test_context_is_capped(self, cfg):
        """More passages is not monotonically better; precision falls off."""
        retr = StubRetriever([[f"Passage {i}" for i in range(40)]])
        llm = ScriptedLLM([answered("Answer.", [])])
        answerer = Answerer(retriever=retr, llm=llm, top_k=40)
        answerer.answer("q?")

        shown = llm.calls[0]["prompt"].count("] source:")
        assert shown <= cfg.max_context_chunks


class TestEvents:
    def test_event_sequence_is_reported_for_streaming(self, cfg):
        retr = StubRetriever([["A"], ["B"]])
        llm = ScriptedLLM([
            insufficient("more"), {"query": "q2"},
            answered("Done [1].", [{"passage": 1, "quote": "A"}]),
        ])
        seen: list[str] = []
        Answerer(retriever=retr, llm=llm).answer("q?", on_event=lambda k, d: seen.append(k))

        assert seen[:4] == ["searching", "retrieved", "reading", "reformulating"]
        assert seen[-2:] == ["answer", "done"]

    def test_abstention_emits_abstained(self, cfg):
        retr = StubRetriever([["A"]])
        llm = ScriptedLLM([insufficient("")])
        seen: list[str] = []
        Answerer(retriever=retr, llm=llm).answer("q?", on_event=lambda k, d: seen.append(k))

        assert "abstained" in seen
