"""Provider routing tests; no network access or real API keys required."""
import importlib.util
import io
import json
import logging
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location("weather_test_server", Path(__file__).with_name("simulation.py"))
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"WEATHERAPI_KEY": "test-weather", "VISUAL_CROSSING_API_KEY": "test-crossing"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        server._WEATHER_CACHE.clear()
        server._WEATHER_COOLDOWNS.clear()
        logging.disable(logging.CRITICAL)
        self.addCleanup(lambda: logging.disable(logging.NOTSET))

    def series(self, days):
        return [server.WeatherData(date=date, temp_c=28, humidity=80, wind_kph=5,
            wind_direction_deg=0, precipitation_mm=2) for date in server._forecast_dates(days)]

    def test_length_routing_and_optional_keys(self):
        for days in (1, 2):
            self.assertEqual(server.weather_provider_order(days), ["WeatherAPI", "Open-Meteo", "Visual Crossing"])
        for days in (3, 7, 14):
            self.assertEqual(server.weather_provider_order(days), ["Open-Meteo", "Visual Crossing"])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(server.weather_provider_order(1), ["Open-Meteo"])

    def test_weatherapi_request_includes_today(self):
        date = server._forecast_dates(1)[0]
        hours = [{"time": f"{date} {h:02d}:00", "temp_c": 28, "humidity": 80,
            "wind_kph": 5, "wind_degree": 359 if h % 2 else 1, "precip_mm": .5} for h in range(24)]
        with patch.object(server, "_weather_json", return_value={"forecast": {"forecastday": [{"date": date, "hour": hours}]}}) as request:
            result = server._fetch_provider("WeatherAPI", 10.7, 122.5, 1)
            self.assertIn("days=2", request.call_args.args[0])
        self.assertEqual(result[0].precipitation_mm, 12)
        self.assertTrue(result[0].wind_direction_deg < 2 or result[0].wind_direction_deg > 358)

    def test_visual_crossing_metric_hourly_contract(self):
        date = server._forecast_dates(1)[0]
        hours = [{"datetime": f"{h:02d}:00:00", "temp": 28, "humidity": 80,
            "windspeed": 5, "winddir": 90, "precip": 0, "source": "fcst"} for h in range(24)]
        data = {"days": [{"datetime": date, "hours": hours, "source": "fcst"}]}
        with patch.object(server, "_weather_json", return_value=data) as request:
            result = server._fetch_provider("Visual Crossing", 10.7, 122.5, 1)
            self.assertIn("unitGroup=metric", request.call_args.args[0])
        self.assertEqual(result[0].wind_kph, 5)
        data["days"][0]["source"] = "stats"
        with patch.object(server, "_weather_json", return_value=data):
            with self.assertRaises(ValueError):
                server._fetch_provider("Visual Crossing", 10.7, 122.5, 1)

    def test_open_meteo_contract(self):
        date = server._forecast_dates(1)[0]
        hourly = {"time": [f"{date}T{h:02d}:00" for h in range(24)]}
        for field, value in (("temperature_2m", 28), ("relative_humidity_2m", 80),
            ("wind_speed_10m", 5), ("wind_direction_10m", 90), ("precipitation", .5)):
            hourly[field] = [value] * 24
        with patch.object(server, "_weather_json", return_value={"hourly": hourly}):
            result = server._fetch_provider("Open-Meteo", 10.7, 122.5, 1)
        self.assertEqual(result[0].precipitation_mm, 12)

    def test_rate_limit_falls_back_and_provider_is_cached(self):
        def fetch(provider, lat, lng, days):
            if provider == "Open-Meteo":
                raise server.ProviderFailure("Daily rate limit reached", 3600)
            return self.series(days)
        with patch.object(server, "_fetch_provider", side_effect=fetch) as provider:
            first = server.fetch_weather(10.7, 122.5, 7)
            second = server.fetch_weather(10.7, 122.5, 3)
            self.assertEqual(provider.call_count, 2)
        self.assertEqual(first[0].provider, "Visual Crossing")
        self.assertTrue(second[0].cached)
        self.assertEqual(len(second), 3)
        self.assertIn("Open-Meteo", server._WEATHER_COOLDOWNS)

    def test_shorter_cache_cannot_satisfy_longer_run(self):
        with patch.object(server, "_fetch_provider", side_effect=lambda p, a, b, d: self.series(d)) as provider:
            server.fetch_weather(10.7, 122.5, 1)
            result = server.fetch_weather(10.7, 122.5, 7)
            self.assertEqual(provider.call_count, 2)
        self.assertEqual(len(result), 7)
        self.assertEqual(result[0].provider, "Open-Meteo")

    def test_incomplete_primary_does_not_contaminate_fallback(self):
        def fetch(provider, lat, lng, days):
            return self.series(days - 1 if provider == "Open-Meteo" else days)
        with patch.object(server, "_fetch_provider", side_effect=fetch):
            result = server.fetch_weather(10.7, 122.5, 3)
        self.assertEqual(len(result), 3)
        self.assertEqual({w.provider for w in result}, {"Visual Crossing"})

    def test_concurrent_cache_and_expiry(self):
        with patch.object(server, "_fetch_provider", side_effect=lambda p, a, b, d: self.series(d)) as provider:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: server.fetch_weather(10.7, 122.5, 7), range(4)))
            self.assertEqual(provider.call_count, 1)
            key = next(iter(server._WEATHER_CACHE))
            server._WEATHER_CACHE[key] = (0, self.series(7))
            server.fetch_weather(10.7, 122.5, 7)
            self.assertEqual(provider.call_count, 2)

    def test_missing_and_invalid_hours_rejected(self):
        rows = [(str(i), 28, 80, 5, 90, 0) for i in range(24)]
        for bad in (rows[:23], [rows[0]] * 24, [(str(i), None, 80, 5, 90, 0) for i in range(24)],
                    [(str(i), 28, 180, 5, 90, 0) for i in range(24)]):
            with self.assertRaises((TypeError, ValueError)):
                server._aggregate_hours("test-date", bad)

    def test_429_honors_retry_after_without_repeating_request(self):
        headers = Message()
        headers["Retry-After"] = "120"
        error = HTTPError("secret-url", 429, "limited", headers, io.BytesIO(json.dumps({"reason": "Daily request limit exceeded"}).encode()))
        with patch.object(server, "urlopen", side_effect=error) as request:
            with self.assertRaises(server.ProviderFailure) as caught:
                server._weather_json("secret-url")
            self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.cooldown, 120)
        self.assertNotIn("secret", str(caught.exception))

    def test_all_failed_providers_are_skipped_during_cooldown(self):
        with patch.object(server, "_fetch_provider", side_effect=server.ProviderFailure("Limited", 300)) as provider:
            for _ in range(2):
                with self.assertRaises(server.HTTPException):
                    server.fetch_weather(10.7, 122.5, 3)
            self.assertEqual(provider.call_count, 2)

    def test_simulation_returns_actual_source(self):
        with patch.object(server, "_fetch_provider", side_effect=lambda p, a, b, d: self.series(d)):
            result = server.simulate(server.SimRequest(days=1, reports=[server.ReportIn(
                id="test", lat=10.7, lng=122.5, status="HIGH", verified=True)]))
        self.assertEqual(result["weather_source"], "WeatherAPI forecast")
        self.assertFalse(result["weather_cached"])
        self.assertEqual(len(result["timeline"]), 1)
        self.assertTrue(result["weather_fetched_at"])


if __name__ == "__main__":
    unittest.main()
