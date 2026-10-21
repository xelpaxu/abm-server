# Weather provider setup

Forecasts start tomorrow in Asia/Manila time. Provider selection:
- Reuse a fresh, complete cached forecast first (30 minutes, per worker).
- 1-2 simulation days: WeatherAPI -> Open-Meteo -> Visual Crossing.
- 3-14 simulation days: Open-Meteo -> Visual Crossing.
- Providers requiring a key are skipped when no key is configured.
- Failed providers have separate cooldowns; one provider supplies every day of a run.
- Cache is held in memory and resets on restart. Multiple worker processes have separate caches.
- No missing hours, statistical forecasts, or fabricated weather are substituted.

WeatherAPI's free three-day window includes today, so it covers two full future simulation days.
The UI's existing weather-source label reports the actual provider and whether the forecast was cached.
Responses also include weather_fetched_at and weather_cached.

## Configure optional free providers

Register at https://www.weatherapi.com/ and https://www.visualcrossing.com/weather-api/.
Keep keys on the Python server. Never put them in NEXT_PUBLIC_ variables or commit .env.

Copy .env.example to .env, enter keys, and start with the explicit environment-file option:

    python -m uvicorn simulation:app --host 0.0.0.0 --port 5000 --env-file .env

Run from this server directory. uvicorn[standard] includes environment-file support.
Alternatively set WEATHERAPI_KEY and VISUAL_CROSSING_API_KEY as server environment variables
and keep using python simulation.py. On hosting, configure environment variables and redeploy.

OPEN_METEO_API_KEY is optional and only for Open-Meteo's paid dedicated endpoint.
No account keys are included with this implementation.

## Verify without spending API quota

    python -B test_weather_providers.py

Alternative-provider adapters are fixture-tested; live WeatherAPI and Visual Crossing calls
require your own keys. Respect each provider's current attribution and usage terms.

Documentation:
- https://www.weatherapi.com/docs/request-parameters/
- https://www.weatherapi.com/pricing
- https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
- https://open-meteo.com/en/docs
