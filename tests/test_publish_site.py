#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

import publish_podcast_asr_site


class PublishSiteTests(unittest.TestCase):
    def test_index_orders_official_publish_time_descending(self) -> None:
        older = {
            "episode_id": "older12345678",
            "title": "Older episode",
            "published_time": "2026-08-01T00:00:00Z",
            "published_at": "2026-08-31T00:00:00",
        }
        newer = {
            "episode_id": "newer12345678",
            "title": "Newer episode",
            "published_time": "2026-08-24T00:00:00Z",
            "published_at": "2026-08-30T00:00:00",
        }

        html = publish_podcast_asr_site.render_site_index([older, newer])

        self.assertLess(html.index("Newer episode"), html.index("Older episode"))

    def test_index_restores_requested_task_before_stale_local_storage(self) -> None:
        html = publish_podcast_asr_site.render_site_index([])

        self.assertIn("const requestedTask=new URLSearchParams(location.search).get('task')", html)
        self.assertIn("if(requestedTask){startPolling(requestedTask);return}", html)
        self.assertIn("const wasActive=activeTask===task.job_id", html)
        self.assertIn("reportIsReady(task)", html)

    def test_index_exposes_global_agent_qa_and_settings(self) -> None:
        html = publish_podcast_asr_site.render_site_index(
            [{"episode_id": "odyssey123456", "title": "Odyssey", "published_time": "2026-09-01"}]
        )

        self.assertEqual(1, html.count('id="podcastQa"'))
        self.assertIn('data-default-scope="all"', html)
        self.assertIn('data-qa-scope="episode"', html)
        self.assertIn('data-qa-scope="all"', html)
        self.assertIn('/api/podcast-asr/qa/questions', html)
        self.assertIn('/static/podcast-asr/settings/index.html', html)

    def test_settings_page_never_contains_an_api_key_value(self) -> None:
        html = publish_podcast_asr_site.render_settings_page()

        self.assertIn('name="api_key"', html)
        self.assertIn('/api/podcast-asr/qa/config', html)
        self.assertNotIn('value="sk-', html)


if __name__ == "__main__":
    unittest.main()
