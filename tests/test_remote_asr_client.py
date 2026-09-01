#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

import remote_asr_client


class RemoteClientTests(unittest.TestCase):
    def test_render_outputs_preserves_complete_chunk_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = remote_asr_client.ChunkResult(
                chunk_index=0,
                start=0.0,
                end=10.0,
                start_ts="00:00:00",
                end_ts="00:00:10",
                duration_seconds=10.0,
                device="remote_cuda",
                model_load_seconds=1.0,
                inference_seconds=0.5,
                wall_seconds=0.7,
                rtf_inference=0.05,
                rtf_wall=0.07,
                chars=4,
                text="测试文本",
                raw_text="测试文本",
                output_json="chunk.json",
                output_txt="chunk.txt",
            )
            output = remote_asr_client.render_outputs(
                {"title": "Test", "chunks": [{"chunk_index": 0}]},
                [result],
                root,
                "remote_cpu",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["device"], "remote_cpu")
            self.assertEqual(payload["summary"]["ok_chunks"], 1)
            self.assertEqual(payload["summary"]["failed_chunks"], 0)
            self.assertIn("测试文本", (root / "transcript_remote_cpu.txt").read_text())


if __name__ == "__main__":
    unittest.main()
