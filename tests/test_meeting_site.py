from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
MEETING_PAGE = APP_ROOT / "web" / "meeting-asr" / "index.html"


class MeetingSiteTests(unittest.TestCase):
    def test_mobile_picker_uses_unrestricted_single_file_input(self) -> None:
        html = MEETING_PAGE.read_text(encoding="utf-8")
        match = re.search(r'<input id="audioFile"[^>]*>', html)

        self.assertIsNotNone(match)
        file_input = match.group(0)
        self.assertNotIn(" accept=", file_input)
        self.assertNotIn(" capture=", file_input)
        self.assertNotIn(" multiple", file_input)
        self.assertIn('id="chooseFileButton"', html)


if __name__ == "__main__":
    unittest.main()
