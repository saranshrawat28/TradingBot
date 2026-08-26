"""
Unit Test Suite for Autonomous AI Trading Daemon and Enhanced Multi-Model Providers.
Tests daemon lifecycle, thought stream queue, provider routing (Groq, Ollama), and auto-execution gates.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.ai.autonomous_daemon import AutonomousAIDaemon
from src.ai.llm_client import LLMClient
from src.engine.ai_guardrails import AIGuardrails
from src.brokers.paper_broker import PaperBroker

class TestAutonomousAIDaemonSuite(unittest.TestCase):
    """
    Test suite verifying AutonomousAIDaemon background engine and thought logging.
    """

    def setUp(self):
        self.daemon = AutonomousAIDaemon.get_instance()
        self.broker = PaperBroker(initial_capital=100000.0)
        self.guardrails = AIGuardrails()

    def tearDown(self):
        if self.daemon.is_active:
            self.daemon.stop()

    def test_daemon_singleton_lifecycle(self):
        """Verify daemon singleton pattern, start, and stop lifecycle."""
        d1 = AutonomousAIDaemon.get_instance()
        d2 = AutonomousAIDaemon.get_instance()
        self.assertIs(d1, d2)
        
        self.assertFalse(self.daemon.is_active)
        mock_llm = MagicMock()
        mock_llm.is_configured.return_value = True
        
        self.daemon.start(
            llm_client=mock_llm,
            guardrails=self.guardrails,
            broker=self.broker,
            is_live_mode=False,
            interval=10
        )
        self.assertTrue(self.daemon.is_active)
        self.assertEqual(self.daemon.scan_interval, 10)
        
        self.daemon.stop()
        self.assertFalse(self.daemon.is_active)

    def test_thought_stream_queue_recording(self):
        """Verify thread-safe in-memory thought queue appending and structure."""
        self.daemon._add_thought("TEST_LEVEL", "Testing thought logging mechanism.", symbol="NIFTY", conviction=8.8)
        thoughts = self.daemon.get_thought_stream()
        self.assertGreater(len(thoughts), 0)
        
        latest = thoughts[0]
        self.assertEqual(latest["level"], "TEST_LEVEL")
        self.assertEqual(latest["symbol"], "NIFTY")
        self.assertEqual(latest["conviction"], 8.8)
        self.assertIn("Testing thought logging", latest["message"])

    def test_llm_client_groq_and_ollama_providers(self):
        """Verify LLMClient initialization and supported provider lists for Groq & Ollama."""
        # Groq Client
        groq_client = LLMClient(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test_key_12345")
        self.assertEqual(groq_client.provider, "groq")
        self.assertEqual(groq_client.model, "llama-3.3-70b-versatile")
        self.assertTrue(groq_client.is_configured())
        self.assertEqual(groq_client.min_call_interval, 0.2)
        
        # Ollama Client (Offline Zero-Key)
        ollama_client = LLMClient(provider="ollama", model="deepseek-r1:latest")
        self.assertEqual(ollama_client.provider, "ollama")
        self.assertTrue(ollama_client.is_configured())

    def test_intraday_squareoff_handling(self):
        """Verify that at 3:15 PM IST the daemon triggers position square-off."""
        # Place a mock position in paper broker
        self.broker.place_order(symbol="NIFTY 24250 CE", quantity=75, side="BUY", price=130.0)
        positions_before = self.broker.get_positions()
        self.assertEqual(len(positions_before), 1)
        
        # Mock get_ist_now to return 15:16 PM (after 3:15 PM)
        with patch("src.ai.autonomous_daemon.get_ist_now") as mock_now:
            fake_time = MagicMock()
            fake_time.hour = 15
            fake_time.minute = 16
            fake_time.strftime.return_value = "15:16:00"
            mock_now.return_value = fake_time
            
            self.daemon._execute_cycle()
            
            positions_after = self.broker.get_positions()
            self.assertEqual(len(positions_after), 0)

if __name__ == "__main__":
    unittest.main()
