import unittest

import app


class RTSPURLNormalizationTests(unittest.TestCase):
    def test_password_is_url_encoded_and_tcp_is_preferred(self):
        raw = "rtsp://admin:Vinu@2710@192.168.100.127:554/cam/realmonitor?channel=2&subtype=0"
        encoded = app.normalize_rtsp_url(raw)
        self.assertIn("admin:Vinu%402710@192.168.100.127:554/cam/realmonitor?channel=2&subtype=0&tcp", encoded)
        self.assertNotIn("Vinu@2710", encoded)


if __name__ == "__main__":
    unittest.main()
