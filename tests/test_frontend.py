"""
Frontend smoke tests using Playwright.
Requires: pip install playwright && playwright install chromium
"""
import os
import sys
import time
import unittest

# Add app directory so the Flask app can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from app import app  # noqa: E402


@unittest.skipUnless(
    os.environ.get("RUN_E2E") == "1",
    "Set RUN_E2E=1 to run Playwright browser tests.",
)
class FlowCardsFrontendTests(unittest.TestCase):
    """Smoke tests that launch a real browser against a local Flask server."""

    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        cls._playwright = sync_playwright().start()
        cls._browser = cls._playwright.chromium.launch(headless=True)

        # Start Flask in a background thread
        import threading
        app.config["TESTING"] = True
        cls._server_thread = threading.Thread(
            target=app.run,
            kwargs={"host": "127.0.0.1", "port": 5199, "debug": False},
            daemon=True,
        )
        cls._server_thread.start()
        time.sleep(1)  # Let the server start

    @classmethod
    def tearDownClass(cls):
        cls._browser.close()
        cls._playwright.stop()

    def setUp(self):
        self.page = self._browser.new_page()

    def tearDown(self):
        self.page.close()

    def test_home_page_loads(self):
        """Verify the app loads with the FlowCards title."""
        self.page.goto("http://127.0.0.1:5199/")
        self.page.wait_for_selector("title")
        title = self.page.title()
        self.assertIn("FlowCards", title)

    def test_views_are_present(self):
        """All six view elements exist in the DOM."""
        self.page.goto("http://127.0.0.1:5199/")
        views = [
            "view-landing",
            "view-home",
            "view-studio",
            "view-flow",
            "view-quiz",
            "view-profile",
        ]
        for vid in views:
            el = self.page.query_selector(f"#{vid}")
            self.assertIsNotNone(el, f"View #{vid} not found in DOM")

    def test_navigate_to_home_after_landing(self):
        """Landing page has a start button that leads to home."""
        self.page.goto("http://127.0.0.1:5199/")
        # Wait for landing load
        self.page.wait_for_selector("#view-landing.active", timeout=5000)
        # Click the get-started button
        start_btn = self.page.query_selector('[onclick*="navigate(\'home\'"]')
        if start_btn:
            start_btn.click()
            self.page.wait_for_timeout(1000)
        home = self.page.query_selector("#view-home.active")
        self.assertIsNotNone(home, "Home view did not become active")

    def test_health_endpoint(self):
        """The /api/health endpoint returns ok:true."""
        import requests as req
        resp = req.get("http://127.0.0.1:5199/api/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])

    def test_dark_mode_theme_attribute(self):
        """The <html> tag has data-theme attribute."""
        self.page.goto("http://127.0.0.1:5199/")
        theme = self.page.get_attribute("html", "data-theme")
        self.assertIsNotNone(theme)
        self.assertIn(theme, ("light", "dark"))


if __name__ == "__main__":
    unittest.main()
