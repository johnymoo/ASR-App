#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda function: function

        post = get
        put = get

    class _HttpException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.APIRouter = _Router
    fastapi_stub.Body = lambda default=..., **kwargs: default
    fastapi_stub.HTTPException = _HttpException
    sys.modules["fastapi"] = fastapi_stub

import podcast_qa_api


class PodcastQaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.saved_paths = {
            name: getattr(podcast_qa_api, name)
            for name in [
                "PODCAST_ROOT",
                "PODCAST_LIBRARY_DIR",
                "QA_ROOT",
                "QA_CONFIG_PATH",
                "QA_TASK_DIR",
                "QA_SESSION_DIR",
            ]
        }
        podcast_qa_api.PODCAST_ROOT = root / "podcast"
        podcast_qa_api.PODCAST_LIBRARY_DIR = root / "static" / "podcast-asr"
        podcast_qa_api.QA_ROOT = root / "qa"
        podcast_qa_api.QA_CONFIG_PATH = root / "qa" / "config.json"
        podcast_qa_api.QA_TASK_DIR = root / "qa" / "tasks"
        podcast_qa_api.QA_SESSION_DIR = root / "qa" / "sessions"
        podcast_qa_api._ensure_dirs()

        output_dir = podcast_qa_api.PODCAST_ROOT / "xiaoyuzhou_odyssey123456" / "output"
        output_dir.mkdir(parents=True)
        transcription = {
            "episode_id": "odyssey123456",
            "title": "《奥德赛》与全球化的第一次崩塌",
            "chunks": [
                {
                    "chunk_index": 0,
                    "start": 1180,
                    "end": 1480,
                    "start_ts": "00:19:40",
                    "end_ts": "00:24:40",
                    "text": (
                        "奥德赛讲的是一个人面对未知旅程的故事。"
                        "千面英雄这本书呢，是我非常想推荐大家去读一读。"
                        "它讨论神话形象如何成为人内心的投射。"
                        "但是这本书读起来并不是特别好读，需要一些神话背景知识。"
                    ),
                    "error": None,
                }
            ],
        }
        (output_dir / "transcription_remote_cpu.json").write_text(
            json.dumps(transcription, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "podcast_summary.json").write_text(
            json.dumps({"title": transcription["title"], "tldr": "从奥德赛讨论历史与人生。"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "episode_page_context.json").write_text(
            json.dumps({"official_title": transcription["title"], "outline": [{"timestamp": "00:20:00", "title": "千面英雄"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "site_meta.json").write_text(
            json.dumps({"slug": "odyssey"}), encoding="utf-8"
        )
        podcast_qa_api.PODCAST_LIBRARY_DIR.mkdir(parents=True)
        (podcast_qa_api.PODCAST_LIBRARY_DIR / "episodes.json").write_text(
            json.dumps([{"episode_id": "odyssey123456", "slug": "odyssey"}]), encoding="utf-8"
        )

    def tearDown(self) -> None:
        for name, value in self.saved_paths.items():
            setattr(podcast_qa_api, name, value)
        self.temp_dir.cleanup()

    def test_retrieval_finds_direct_book_recommendation(self) -> None:
        episode = podcast_qa_api.discover_episodes()[0]
        records = podcast_qa_api.build_records(episode)

        results = podcast_qa_api.retrieve(records, "里面推荐了一本书，书名是什么？")

        transcript_results = [item for item in results if item["source_type"] == "transcript"]
        self.assertTrue(transcript_results)
        self.assertIn("千面英雄", transcript_results[0]["text"])
        self.assertEqual("00:19:40", transcript_results[0]["timestamp"])
        self.assertTrue(transcript_results[0]["url"].endswith("/odyssey/full.html#chunk-000"))

    def test_context_includes_summary_outline_and_transcript(self) -> None:
        evidence, _ = podcast_qa_api.assemble_context(
            podcast_qa_api.discover_episodes(), "推荐的书", "all", None
        )

        self.assertEqual({"summary", "outline", "transcript"}, {item["source_type"] for item in evidence})

    def test_follow_up_retrieval_uses_previous_answer_not_previous_question(self) -> None:
        history = [
            {"role": "user", "content": "里面推荐了一本书，书名是什么？"},
            {"role": "assistant", "content": "《千面英雄》"},
        ]
        query = podcast_qa_api._retrieval_query("节目说这本书好读吗？", history)
        records = podcast_qa_api.build_records(podcast_qa_api.discover_episodes()[0])

        results, _ = podcast_qa_api.assemble_context(
            podcast_qa_api.discover_episodes(), ["节目说这本书好读吗？", query], "episode", "odyssey123456"
        )

        self.assertNotIn("推荐", query)
        self.assertTrue(any("并不是特别好读" in item["text"] for item in results[:5]))

    def test_scope_change_keeps_session_but_resets_model_history(self) -> None:
        session = {
            "scope": "episode",
            "episode_id": "odyssey123456",
            "messages": [
                {"role": "user", "content": "推荐了哪本书？"},
                {"role": "assistant", "content": "《千面英雄》"},
            ],
        }

        history = podcast_qa_api._history_for_task(
            session, {"scope": "all", "episode_id": None}
        )

        self.assertEqual([], history)
        self.assertEqual(2, len(session["messages"]))

    def test_public_config_masks_api_key(self) -> None:
        public = podcast_qa_api.public_config(
            {
                "api_base": "https://api.openai.com/v1",
                "api_key": "secret-example-key",
                "model": "gpt-5",
                "system_prompt": "Use evidence",
                "temperature": 0.1,
                "timeout_seconds": 180,
            }
        )

        self.assertNotIn("api_key", public)
        self.assertTrue(public["api_key_set"])
        self.assertTrue(public["api_key_masked"].endswith("-key"))
        self.assertNotIn("secret-example-key", json.dumps(public))

    def test_global_episode_selection_prioritizes_exact_title_term(self) -> None:
        records = [
            {
                "evidence_id": "generic:episode",
                "episode_id": "generic",
                "episode_title": "AI 与宏观经济",
                "source_type": "episode",
                "timestamp": "",
                "text": "这期播客讨论了人工智能和市场。",
                "url": "#",
            },
            {
                "evidence_id": "spacex:episode",
                "episode_id": "spacex",
                "episode_title": "口述 SpaceX 开发史",
                "source_type": "episode",
                "timestamp": "",
                "text": "前高管回顾火箭制造和公司发展。",
                "url": "#",
            },
        ]

        result = podcast_qa_api.retrieve(records, "哪些播客讨论了 SpaceX？")

        self.assertEqual("spacex", result[0]["episode_id"])

    def test_async_question_persists_follow_up_session(self) -> None:
        fake_result = {
            "answer": "推荐的是《千面英雄》。",
            "citations": [
                {
                    "episode_id": "odyssey123456",
                    "episode_title": "《奥德赛》与全球化的第一次崩塌",
                    "source_type": "transcript",
                    "timestamp": "00:19:40",
                    "text": "千面英雄这本书呢，是我非常想推荐大家去读一读。",
                    "url": "/static/podcast-asr/odyssey/full.html#chunk-000",
                }
            ],
        }
        with mock.patch.object(podcast_qa_api, "answer_question", return_value=fake_result):
            created = podcast_qa_api.create_question(
                {"question": "推荐了哪本书？", "scope": "episode", "episode_id": "odyssey123456"}
            )
            task = created["task"]
            for _ in range(100):
                state = podcast_qa_api._read_json(podcast_qa_api._task_path(task["task_id"]), {})
                if state.get("status") in podcast_qa_api._TERMINAL_STATES:
                    break
                time.sleep(0.01)

        self.assertEqual("completed", state["status"])
        session = podcast_qa_api.get_session(task["session_id"])["session"]
        self.assertEqual(["user", "assistant"], [item["role"] for item in session["messages"]])
        self.assertEqual("推荐的是《千面英雄》。", session["messages"][1]["content"])
        self.assertEqual("episode", session["scope"])


if __name__ == "__main__":
    unittest.main()
