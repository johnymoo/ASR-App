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


if __name__ == "__main__":
    unittest.main()
