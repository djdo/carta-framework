import unittest
from unittest.mock import MagicMock, patch
from carta_framework.chat_engine import ChatEngine
from carta_framework.models import TemporalScope, AgenticWorkflow
from carta_framework.vector_store import BaseVectorStore

class TestChatEngine(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock(spec=BaseVectorStore)
        
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = "This is a mock answer."
        
        self.engine = ChatEngine(
            llm_provider=self.mock_llm,
            vector_store=self.mock_vector_store,
            system_prompt="Test prompt"
        )

    def test_answer_question_propagation(self):
        self.mock_vector_store.query.return_value = {
            "documents": ["context doc"],
            "metadatas": [{}],
            "distances": [0.1]
        }
        
        scope = TemporalScope(horizon_timestamp=2000.0)
        res = self.engine.answer_question(
            question="What?",
            history=[],
            tenant_id="test_tenant",
            temporal_scope=scope
        )
        
        # Verify VectorStore query received exact tenant and scope parameters
        self.mock_vector_store.query.assert_called_once()
        kwargs = self.mock_vector_store.query.call_args[1]
        self.assertEqual(kwargs["tenant_id"], "test_tenant")
        self.assertEqual(kwargs["temporal_scope"], scope)
        self.assertEqual(res["answer"], "This is a mock answer.")

    def test_workflow_and_personality_injection(self):
        from carta_framework.models import AgenticWorkflow
        
        self.mock_vector_store.query.return_value = {
            "documents": ["context doc"],
            "metadatas": [{}],
            "distances": [0.1]
        }
        
        workflow = AgenticWorkflow(
            id="test_wf",
            name="Test",
            description="A test workflow",
            system_prompt_extension="Generate JSON."
        )
        
        self.engine.answer_question(
            question="What?",
            system_prompt_override="Overridden Default.",
            workflow=workflow
        )
        
        self.mock_llm.chat.assert_called_once()
        messages = self.mock_llm.chat.call_args[1].get("messages") or self.mock_llm.chat.call_args[0][0]
        
        sys_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        
        self.assertIn("Overridden Default.", sys_message)
        self.assertNotIn("Test prompt", sys_message)
        self.assertIn("=== EXPLICIT INSTRUCTION OVERRIDE ===", sys_message)
        self.assertIn("Generate JSON.", sys_message)

        # Follow-up #25: the workflow is restated after the question, with a neutral final line
        user_message = messages[-1]["content"]
        self.assertTrue(user_message.endswith("Use the context above when responding.\n\n[Reminder] Generate JSON."))
        self.assertNotIn("Answer the question based on", user_message)

    def test_workflow_reminder_defaults_to_first_two_sentences(self):
        workflow = AgenticWorkflow(id="w", name="W", system_prompt_extension="One. Two! Three? Four.")
        self.assertEqual(workflow.reminder_text(), "One. Two!")
        self.assertEqual(AgenticWorkflow(id="w", name="W", system_prompt_extension="x", reminder="Short.").reminder_text(), "Short.")

    def test_stream_event_names(self):
        self.mock_vector_store.query.return_value = {
            "documents": ["context doc"],
            "metadatas": [{"source": "a.pdf"}],
            "distances": [0.1]
        }
        self.mock_llm.chat.return_value = iter(["Hel", "lo"])

        events = list(self.engine.answer_question_stream(question="What?"))

        self.assertTrue(events[0].startswith("event: sources\n"))
        self.assertEqual(events[1:3], ['event: message\ndata: "Hel"\n\n', 'event: message\ndata: "lo"\n\n'])
        self.assertEqual(events[-1], "event: end\ndata: [DONE]\n\n")

    def test_context_blocks_carry_delimiters_and_provenance(self):
        context = self.engine._build_context(
            ["Source: Lecture10_Timers.pdf (Page 12/40)\n\nTimer text", "Plain text"],
            [{"source": "Lecture10_Timers.pdf", "page_number": 12}, {"source": "lab.s"}]
        )

        self.assertIn("--- Context Document [1] (Source: Lecture10_Timers.pdf, Page 12) ---\nTimer text\n" + "-" * 62, context)
        self.assertIn("--- Context Document [2] (Source: lab.s) ---\nPlain text\n" + "-" * 62, context)
        self.assertNotIn("(Page 12/40)", context)

    def test_reply_cap_is_passed_to_provider(self):
        self.mock_vector_store.query.return_value = {"documents": ["ctx"], "metadatas": [{}], "distances": [0.1]}
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt", max_reply_tokens=600)

        engine.answer_question(question="What?")

        self.assertEqual(self.mock_llm.chat.call_args[1]["max_tokens"], 600)

    def test_history_is_windowed_then_trimmed_to_context_window(self):
        self.mock_vector_store.query.return_value = {"documents": ["ctx"], "metadatas": [{}], "distances": [0.1]}
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * 400} for i in range(10)]

        # Unlimited window: only the last 6 messages are kept.
        self.engine.answer_question(question="What?", history=history)
        messages = self.mock_llm.chat.call_args[0][0]
        self.assertEqual([m["content"][:3] for m in messages[1:-1]], ["m4 ", "m5 ", "m6 ", "m7 ", "m8 ", "m9 "])

        # Tight window: the oldest messages are dropped until the request fits.
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt",
                            max_reply_tokens=100, context_window=400)
        engine.answer_question(question="What?", history=history)
        messages = self.mock_llm.chat.call_args[0][0]
        kept = messages[1:-1]
        self.assertLess(len(kept), 6)
        self.assertEqual(kept[-1]["content"][:3], "m9 ")

    def test_sources_carry_filename_and_similarity_score(self):
        self.mock_vector_store.query.return_value = {
            "documents": ["Timer text"], "metadatas": [{"source": "Lecture10.pdf", "page_number": 3}], "distances": [0.2]
        }
        source = self.engine.answer_question(question="What?")["sources"][0]
        self.assertEqual(source.filename, "Lecture10.pdf")
        self.assertEqual(source.similarity_score, 0.9)

    def test_usage_is_reported_for_full_request(self):
        self.mock_vector_store.query.return_value = {"documents": ["ctx " * 100], "metadatas": [{}], "distances": [0.1]}
        self.mock_llm.chat.return_value = iter(["ab" * 40])
        usage = {}
        list(self.engine.answer_question_stream(question="What?", usage=usage))
        # The prompt estimate covers the system prompt, context and question, not just the question.
        self.assertGreater(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 21)

    def test_no_relevant_chunk_returns_no_answer_response_without_model_call(self):
        # d = 1.5 gives s = 0.25, below the default threshold of 0.3
        self.mock_vector_store.query.return_value = {"documents": ["far"], "metadatas": [{}], "distances": [1.5]}
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt", no_answer_response="Not covered.")

        res = engine.answer_question(question="What?")
        events = list(engine.answer_question_stream(question="What?"))

        self.assertEqual(res, {"answer": "Not covered.", "sources": []})
        self.assertEqual(events, ["event: sources\ndata: []\n\n", 'event: message\ndata: "Not covered."\n\n',
                                  "event: end\ndata: [DONE]\n\n"])
        self.mock_llm.chat.assert_not_called()

    def test_no_answer_response_none_sends_empty_context_to_model(self):
        self.mock_vector_store.query.return_value = {"documents": [], "metadatas": [], "distances": []}
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt", no_answer_response=None)

        res = engine.answer_question(question="What?")

        self.assertEqual(res["answer"], "This is a mock answer.")
        self.mock_llm.chat.assert_called_once()

    def test_session_history_is_read_from_and_saved_to_state_manager(self):
        from carta_framework.state import MemoryStateManager
        self.mock_vector_store.query.return_value = {"documents": ["ctx"], "metadatas": [{}], "distances": [0.1]}
        state = MemoryStateManager()
        for i in range(8):
            state.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt", state_manager=state)

        engine.answer_question(question="Q1", session_id="s1")
        messages = self.mock_llm.chat.call_args[0][0]
        self.assertEqual([m["content"] for m in messages[1:-1]], ["m2", "m3", "m4", "m5", "m6", "m7"])
        self.assertEqual(state.get_history("s1", limit=2),
                         [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "This is a mock answer."}])

        # Streaming saves the joined answer once the stream ends; an explicit history wins.
        self.mock_llm.chat.return_value = iter(["Hel", "lo"])
        list(engine.answer_question_stream(question="Q2", history=[], session_id="s1"))
        self.assertEqual(len(self.mock_llm.chat.call_args[0][0]), 2)
        self.assertEqual(state.get_history("s1", limit=1), [{"role": "assistant", "content": "Hello"}])

    def test_without_session_id_state_manager_is_untouched(self):
        state = MagicMock()
        self.mock_vector_store.query.return_value = {"documents": ["ctx"], "metadatas": [{}], "distances": [0.1]}
        engine = ChatEngine(self.mock_llm, self.mock_vector_store, "Test prompt", state_manager=state)

        engine.answer_question(question="What?", history=[])

        state.get_history.assert_not_called()
        state.add_message.assert_not_called()


class TestMemoryStateManager(unittest.TestCase):
    def test_listing_interface(self):
        from carta_framework.state import MemoryStateManager
        sm = MemoryStateManager()
        for i in range(8):
            sm.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        self.assertEqual([m["content"] for m in sm.get_history("s1")], ["m2", "m3", "m4", "m5", "m6", "m7"])
        self.assertEqual(sm.get_history("s1", limit=2), [{"role": "user", "content": "m6"}, {"role": "assistant", "content": "m7"}])
        sm.clear_history("s1")
        self.assertEqual(sm.get_history("s1"), [])
