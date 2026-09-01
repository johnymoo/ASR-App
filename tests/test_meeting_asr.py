import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))
sys.path.insert(0, str(APP_ROOT / "services" / "x570-asr"))

import meeting_asr_pipeline

try:
    import worker
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    worker = None


class MeetingTranscriptTests(unittest.TestCase):
    def test_user_facing_transcript_format(self):
        text = meeting_asr_pipeline._format_transcript(
            [
                {"speaker": 0, "start": 0, "text": "说到昆山嘛，"},
                {"speaker": 1, "start": 2.8, "text": "嗯，目前情况是这样。"},
            ]
        )
        self.assertEqual(
            text,
            "发言人 1 0:00:00\n说到昆山嘛，\n\n发言人 2 0:00:02\n嗯，目前情况是这样。\n",
        )

@unittest.skipIf(worker is None, "torch is only installed in the ASR runtime")
class MeetingWorkerTests(unittest.TestCase):
    def test_split_and_merge_long_speaker_turn(self):
        pieces = worker._split_speaker_timeline([[0.0, 60.0, 1]])
        self.assertEqual(len(pieces), 3)
        merged = worker._merge_speaker_segments(
            [{**piece, "text": f"第{index}段。"} for index, piece in enumerate(pieces)]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["speaker"], 1)
        self.assertEqual(merged[0]["text"], "第0段。第1段。第2段。")

    def test_clean_sensevoice_tags(self):
        self.assertEqual(worker._clean_text("<|zh|><|Speech|> 你好  世界"), "你好 世界")


if __name__ == "__main__":
    unittest.main()
