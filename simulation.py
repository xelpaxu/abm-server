"""Weather-driven Aedes inspection scenarios; see docs/SIMULATION_RRL.md.

One step is one calendar day. Agents are scenario units, not measured mosquitoes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
import random
import logging
import time
from threading import Lock
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

LOCAL_TZ = timezone(timedelta(hours=8))
MODEL_VERSION = "2.0.0"
CLOSED = {"RESOLVED", "COMPLETED", "CLOSED", "DISMISSED", "REJECTED"}
SEVERITY = {"CRITICAL": 1.0, "HIGH": 0.8, "MODERATE": 0.5, "MEDIUM": 0.5,
            "LOW": 0.2, "SAFE": 0.1, "OPEN": 0.5, "PENDING": 0.5,
            "VERIFIED": 0.5, "UNVERIFIED": 0.5, "IN PROGRESS": 0.5, "ASSIGNED": 0.5}
# Reports stored before the risk vocabulary changed to CRITICAL/HIGH/MODERATE/LOW
# carry the older display labels. Map them rather than rejecting the report.
LEGACY_STATUS = {"HIGH RISK": "HIGH", "LOW RISK": "LOW", "MODERATE RISK": "MODERATE"}
LIMITATIONS = [
    "Inspection scenario only: scores are relative priorities, not dengue probabilities or measured mosquito counts.",
    "Movement, seeding, survival and reproduction coefficients need local trap-data calibration; references support mechanisms, not these exact forecasts.",
    "Weather is a regional model forecast at the report centroid, not street-level observations. Longer horizons are less certain.",
    "Wind adds a weak downwind scenario bias; buildings, hosts, drainage, terrain and actual water availability are not mapped.",
    "Unresolved sites are assumed to persist without cleaning. Inferred locations require field inspection.",
]
REFERENCES = [
    {"title": "CDC: Life Cycle of Aedes Mosquitoes", "url": "https://www.cdc.gov/mosquitoes/about/life-cycle-of-aedes-mosquitoes.html"},
    {"title": "CDC: About Mosquitoes", "url": "https://www.cdc.gov/mosquitoes/about/index.html"},
    {"title": "Brady et al. (2013): temperature-dependent adult survival", "url": "https://doi.org/10.1186/1756-3305-6-351"},
    {"title": "Mark-release-recapture of male Aedes aegypti (2021)", "url": "https://doi.org/10.1371/journal.pntd.0009357"},
    {"title": "Open-Meteo forecast documentation", "url": "https://open-meteo.com/en/docs"},
]


class Schema(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class ReportIn(Schema):
    id: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-85, le=85)
    lng: float = Field(ge=-180, le=180)
    status: str = "MODERATE"
    verified: bool = False
    accuracy: float = Field(default=85, ge=0, le=100)
    resolved: bool = False
    resolved_at: float | None = None
    location_name: str = Field(default="Reported site", max_length=200)

    @field_validator("status")
    @classmethod
    def normalize_status(cls, value):
        value = value.strip().upper()
        value = LEGACY_STATUS.get(value, value)
        if value not in SEVERITY and value not in CLOSED:
            raise ValueError("Unknown report status")
        return value


class WeatherData(Schema):
    temp_c: float = Field(ge=-60, le=60)
    humidity: float = Field(ge=0, le=100)
    wind_kph: float = Field(ge=0, le=250)
    wind_direction_deg: float = Field(ge=0, le=360)
    precipitation_mm: float = Field(ge=0, le=1000)
    condition: str = "Forecast"
    date: str = ""
    provider: str = ""
    fetched_at: str = ""
    cached: bool = False


class SimRequest(Schema):
    days: int = Field(default=3, ge=1, le=14)
    reports: list[ReportIn] = Field(default_factory=list, max_length=500)
    scope: str = Field(default="all_unresolved", pattern="^(all_unresolved|high_risk)$")
    seed: int = Field(default=42, ge=0, le=2147483647)
    weather_override: WeatherData | None = None


def distance_m(lat1, lng1, lat2, lng2):
    north = (lat2 - lat1) * 111320
    east = (lng2 - lng1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(north, east)



# Cache full validated sequences; metadata stays attached to the chosen provider.
_WEATHER_CACHE = {}
_WEATHER_LOCK = Lock()
_WEATHER_TTL_SECONDS = 1800
_WEATHER_COOLDOWNS = {}


class ProviderFailure(Exception):
    def __init__(self, message, cooldown=60):
        super().__init__(message)
        self.cooldown = cooldown


def weather_provider_order(days):
    # Free WeatherAPI includes today, leaving two full days from tomorrow.
    providers = []
    if days <= 2 and os.getenv("WEATHERAPI_KEY", "").strip():
        providers.append("WeatherAPI")
    providers.append("Open-Meteo")
    if os.getenv("VISUAL_CROSSING_API_KEY", "").strip():
        providers.append("Visual Crossing")
    return providers


def _retry_delay(headers, reason):
    delay = 3600 if "hour" in reason.lower() else 60
    if "daily" in reason.lower() or "day" in reason.lower():
        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        delay = max(60, math.ceil((midnight - now).total_seconds()))
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            try:
                return max(1, math.ceil(parsedate_to_datetime(value).timestamp() - time.time()))
            except (TypeError, ValueError, OverflowError):
                pass
    return delay


def _weather_json(url):
    # One bounded attempt per provider; fail over instead of multiplying retries.
    try:
        with urlopen(url, timeout=8) as response:
            data = json.load(response)
        if not isinstance(data, dict):
            raise ProviderFailure("Invalid forecast response")
        if data.get("error"):
            error = data["error"]
            code = error.get("code") if isinstance(error, dict) else None
            if code == 2007:
                raise ProviderFailure("Monthly quota exceeded", 3600)
            if code in (1002, 2006, 2008, 2009):
                raise ProviderFailure("API key missing, invalid or not authorized for this forecast", 300)
            raise ProviderFailure("Provider rejected the forecast request")
        return data
    except HTTPError as exc:
        reason = ""
        try:
            body = json.loads(exc.read())
            error = body.get("error", {})
            reason = str(body.get("reason") or (error.get("message") if isinstance(error, dict) else ""))[:300]
            code = error.get("code") if isinstance(error, dict) else None
        except (ValueError, OSError, AttributeError):
            code = None
        if exc.code == 429 or code == 2007:
            # Do not return provider URLs/messages that might echo credentials.
            quota = "daily" if "daily" in reason.lower() else "hourly" if "hour" in reason.lower() else "request"
            raise ProviderFailure(f"{quota.capitalize()} rate limit reached", _retry_delay(exc.headers, reason)) from None
        if exc.code in (401, 403) or code in (1002, 2006, 2008, 2009):
            raise ProviderFailure("API key invalid or plan does not allow this forecast", 300) from None
        raise ProviderFailure(f"Provider HTTP {exc.code}", 60) from None
    except (OSError, TimeoutError):
        raise ProviderFailure("Connection failed or timed out", 60) from None
    except (ValueError, TypeError):
        raise ProviderFailure("Invalid forecast JSON", 60) from None


def _forecast_dates(days):
    today = datetime.now(LOCAL_TZ).date()
    return [(today + timedelta(days=i)).isoformat() for i in range(1, days + 1)]


def _aggregate_hours(date, rows):
    if len(rows) != 24 or len({row[0] for row in rows}) != 24:
        raise ValueError(f"Expected 24 distinct hourly forecasts for {date}")
    # Row format: timestamp, temperature C, humidity %, wind km/h, direction, rain mm.
    values = [[float(v) for v in row[1:]] for row in rows]
    if any(len(v) != 5 or not all(math.isfinite(x) for x in v) for v in values):
        raise ValueError("Missing/non-finite hourly weather values")
    if any(not (-60 <= v[0] <= 60 and 0 <= v[1] <= 100 and 0 <= v[2] <= 250
                and 0 <= v[3] <= 360 and 0 <= v[4] <= 1000) for v in values):
        raise ValueError("Hourly weather values out of range")
    east = sum(v[2] * math.sin(math.radians(v[3])) for v in values)
    north = sum(v[2] * math.cos(math.radians(v[3])) for v in values)
    rain = sum(v[4] for v in values)
    return WeatherData(date=date, temp_c=round(sum(v[0] for v in values) / 24, 2),
        humidity=round(sum(v[1] for v in values) / 24, 2),
        wind_kph=round(sum(v[2] for v in values) / 24, 2),
        wind_direction_deg=round(math.degrees(math.atan2(east, north)) % 360, 2),
        precipitation_mm=round(rain, 2),
        condition="Rain forecast" if rain >= 1 else "Little or no rain forecast")


def _fetch_provider(provider, lat, lng, days):
    dates = _forecast_dates(days)
    if provider == "Open-Meteo":
        parameters = {"latitude": lat, "longitude": lng, "timezone": "Asia/Manila",
            "forecast_days": days + 1, "wind_speed_unit": "kmh",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation"}
        key = os.getenv("OPEN_METEO_API_KEY", "").strip()
        endpoint = "https://api.open-meteo.com/v1/forecast"
        if key:
            parameters["apikey"] = key
            endpoint = "https://customer-api.open-meteo.com/v1/forecast"
        hourly = _weather_json(endpoint + "?" + urlencode(parameters))["hourly"]
        result = []
        for date in dates:
            rows = [(stamp, *(hourly[field][i] for field in (
                "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "precipitation")))
                for i, stamp in enumerate(hourly["time"]) if stamp.startswith(date)]
            result.append(_aggregate_hours(date, rows))
        return result
    if provider == "WeatherAPI":
        if days > 2:
            raise ValueError("Free WeatherAPI does not cover this many full future days")
        parameters = {"key": os.environ["WEATHERAPI_KEY"], "q": f"{lat},{lng}",
            "days": days + 1, "aqi": "no", "alerts": "no"}
        data = _weather_json("https://api.weatherapi.com/v1/forecast.json?" + urlencode(parameters))
        records = {day["date"]: day for day in data["forecast"]["forecastday"]}
        return [_aggregate_hours(date, [(h["time"], h["temp_c"], h["humidity"], h["wind_kph"],
            h["wind_degree"], h["precip_mm"]) for h in records[date]["hour"]]) for date in dates]
    if provider == "Visual Crossing":
        # A forecast-only Timeline query; no historical/statistical weather substitution.
        parameters = {"key": os.environ["VISUAL_CROSSING_API_KEY"], "unitGroup": "metric",
            "include": "days,hours", "contentType": "json",
            "elements": "datetime,temp,humidity,windspeed,winddir,precip,source"}
        data = _weather_json(
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lng}?"
            + urlencode(parameters))
        records = {day["datetime"]: day for day in data["days"]}
        result = []
        for date in dates:
            day = records[date]
            if day.get("source") not in (None, "fcst"):
                raise ValueError("Expected forecast weather, not observations or climatology")
            rows = []
            for hour in day["hours"]:
                if hour.get("source") not in (None, "fcst"):
                    raise ValueError("Expected forecast hourly data")
                rows.append((hour["datetime"], hour["temp"], hour["humidity"],
                    hour["windspeed"], hour["winddir"], hour["precip"]))
            result.append(_aggregate_hours(date, rows))
        return result
    raise ValueError("Unknown weather provider")


def fetch_weather(lat: float, lng: float, days: int) -> list[WeatherData]:
    """Select a compatible provider, cache valid results, and skip failed providers."""
    if not 1 <= days <= 14:
        raise HTTPException(422, "Forecast duration must be between 1 and 14 days")
    # Waiting clients reuse the successful result rather than duplicating requests.
    if not _WEATHER_LOCK.acquire(timeout=10):
        raise HTTPException(503, "Another forecast is still loading. Please retry shortly.")
    try:
        today = datetime.now(LOCAL_TZ).date().isoformat()
        location = (round(lat, 4), round(lng, 4), today)
        providers = weather_provider_order(days)
        now = time.time()
        for key, (created, _) in list(_WEATHER_CACHE.items()):
            if now - created >= _WEATHER_TTL_SECONDS or key[2] != today:
                del _WEATHER_CACHE[key]
        # Any complete, fresh cached sequence beats another external request.
        for provider in providers:
            cached = _WEATHER_CACHE.get((*location, provider))
            if cached and len(cached[1]) >= days:
                return [w.model_copy(update={"cached": True}) for w in cached[1][:days]]
        errors = []
        for provider in providers:
            blocked_until, reason = _WEATHER_COOLDOWNS.get(provider, (0, ""))
            remaining = math.ceil(blocked_until - time.time())
            if remaining > 0:
                errors.append(f"{provider}: {reason}; retry in {remaining}s")
                continue
            try:
                forecast = _fetch_provider(provider, lat, lng, days)
                if len(forecast) != days or [w.date for w in forecast] != _forecast_dates(days):
                    raise ValueError("Forecast does not cover the requested dates")
            except (ProviderFailure, ValueError, KeyError, TypeError, IndexError) as exc:
                safe_reason = str(exc) if isinstance(exc, ProviderFailure) else "Incomplete or invalid hourly forecast"
                delay = exc.cooldown if isinstance(exc, ProviderFailure) else 60
                _WEATHER_COOLDOWNS[provider] = (time.time() + delay, safe_reason)
                logging.getLogger(__name__).warning("%s forecast failed: %s", provider, safe_reason)
                errors.append(f"{provider}: {safe_reason}; retry in {delay}s")
                continue
            fetched = datetime.now(timezone.utc).isoformat()
            forecast = [w.model_copy(update={"provider": provider, "fetched_at": fetched, "cached": False}) for w in forecast]
            if len(_WEATHER_CACHE) >= 128:
                del _WEATHER_CACHE[min(_WEATHER_CACHE, key=lambda k: _WEATHER_CACHE[k][0])]
            _WEATHER_CACHE[(*location, provider)] = (time.time(), [w.model_copy() for w in forecast])
            _WEATHER_COOLDOWNS.pop(provider, None)
            return forecast
        missing = []
        if days <= 2 and not os.getenv("WEATHERAPI_KEY", "").strip():
            missing.append("WEATHERAPI_KEY")
        if not os.getenv("VISUAL_CROSSING_API_KEY", "").strip():
            missing.append("VISUAL_CROSSING_API_KEY")
        detail = "No configured weather provider could supply the full forecast. " + "; ".join(errors)
        if missing:
            detail += ". Enable additional free providers with server environment variables: " + ", ".join(missing)
        raise HTTPException(503, detail)
    finally:
        _WEATHER_LOCK.release()


def suitability(w: WeatherData):
    # Illustrative thermal response, NOT a fitted published equation.
    return max(0.0, 1 - ((w.temp_c - 28) / 14) ** 2) * (0.4 + 0.6 * w.humidity / 100)


def daily_survival(age: int, w: WeatherData):
    # Weibull conditional survival: mean life ~21 days without added stress.
    age_hazard = ((age + 1) / 24) ** 2 - (age / 24) ** 2
    stress = max(0, abs(w.temp_c - 27) - 5) * 0.015 + max(0, 60 - w.humidity) * 0.002
    return math.exp(-age_hazard - stress)


@dataclass
class MosquitoAgent:
    id: int
    source: ReportIn
    lat: float
    lng: float
    age: int
    score: float


@dataclass
class Immature:
    source: ReportIn
    progress: float = 0


class MosquitoModel:
    """Bounded model independent of Mesa scheduler API versions."""
    MAX_AGENTS = 6000

    def __init__(self, reports, seed):
        self.rng = random.Random(seed)
        self.reports = reports
        self.agents = []
        self.immatures = []
        self.next_id = 0
        self.capped = False
        for report in reports:
            weight = SEVERITY[report.status] * report.accuracy / 100 * (1 if report.verified else 0.6)
            for _ in range(round(12 * weight)):
                self.add_adult(report, self.rng.randrange(0, 15))
            for _ in range(round(8 * weight)):
                self.immatures.append(Immature(report))

    def add_adult(self, report, age=0):
        if len(self.agents) >= self.MAX_AGENTS:
            self.capped = True
            return
        self.next_id += 1
        self.agents.append(MosquitoAgent(self.next_id, report, report.lat, report.lng, age,
                                        SEVERITY[report.status] * report.accuracy / 100))

    def step(self, weather):
        survivors, deaths, births = [], 0, 0
        activity = suitability(weather) / (1 + weather.wind_kph / 12) / (1 + weather.precipitation_mm / 20)
        # Meteorological wind direction is FROM; reverse it for downwind drift.
        downwind = math.radians((weather.wind_direction_deg + 180) % 360)
        for agent in self.agents:
            if self.rng.random() > daily_survival(agent.age, weather):
                deaths += 1
                continue
            agent.age += 1
            angle = self.rng.uniform(0, 2 * math.pi)
            flight = self.rng.uniform(15, 100) * activity
            drift = min(15, weather.wind_kph * 0.8) * activity
            north = flight * math.cos(angle) + drift * math.cos(downwind)
            east = flight * math.sin(angle) + drift * math.sin(downwind)
            # Weak fidelity to the originating habitat proxy.
            north += (agent.source.lat - agent.lat) * 111320 * 0.08
            east += (agent.source.lng - agent.lng) * 111320 * math.cos(math.radians(agent.lat)) * 0.08
            agent.lat += north / 111320
            agent.lng += east / (111320 * math.cos(math.radians(agent.lat)))
            survivors.append(agent)
        self.agents = survivors.copy()
        pending = []
        development_days = max(7, min(21, 10 - (weather.temp_c - 25) * 0.5))
        for immature in self.immatures:
            survival = 0.96 if 15 <= weather.temp_c <= 35 else 0.7
            if self.rng.random() > survival:
                continue
            if weather.temp_c >= 15:
                immature.progress += 1 / development_days
            if immature.progress >= 1 - 1e-9:
                before = len(self.agents)
                self.add_adult(immature.source)
                births += len(self.agents) - before
            else:
                pending.append(immature)
        self.immatures = pending
        # Deposits only at reported sites; new adults cannot reproduce today.
        for agent in survivors:
            if agent.age < 3 or distance_m(agent.lat, agent.lng, agent.source.lat, agent.source.lng) > 150:
                continue
            chance = 0.18 * suitability(weather) * (1 + min(weather.precipitation_mm, 20) / 40)
            if self.rng.random() < chance:
                if len(self.immatures) < self.MAX_AGENTS:
                    self.immatures.append(Immature(agent.source))
                else:
                    self.capped = True
        return deaths, births


def predictions(model, weather):
    # ~100m grid, no fabricated administrative bounds or place names.
    cells = {}
    if not model.reports:
        return []
    origin = model.reports[0]
    cos_lat = math.cos(math.radians(origin.lat))
    for agent in model.agents:
        key = (math.floor((agent.lat - origin.lat) * 111320 / 100),
               math.floor((agent.lng - origin.lng) * 111320 * cos_lat / 100))
        cells.setdefault(key, []).append(agent)
    result = []
    for members in cells.values():
        lat = sum(a.lat for a in members) / len(members)
        lng = sum(a.lng for a in members) / len(members)
        nearest = min(model.reports, key=lambda r: distance_m(lat, lng, r.lat, r.lng))
        score = min(1, 0.5 * (1 - math.exp(-len(members) / 12)) +
                    0.35 * sum(a.score for a in members) / len(members) + 0.15 * suitability(weather))
        result.append({"lat": round(lat, 6), "lng": round(lng, 6), "risk_score": round(score, 3),
            "risk_level": "CRITICAL" if score >= 0.8 else "HIGH" if score >= 0.55 else "MODERATE",
            "agent_count": len(members), "inspection_radius_m": 100,
            "source_report_ids": sorted({a.source.id for a in members}),
            "location_estimate": f"Near {nearest.location_name} ({distance_m(lat, lng, nearest.lat, nearest.lng):.0f} m)",
            "reasoning": f"{len(members)} simulated adult units in a 100 m grid cell. Inspect accessible water-holding containers within 100 m; confirm habitat on site. Relative priority only."})
    return sorted(result, key=lambda p: (-p["risk_score"], -p["agent_count"]))[:20]


app = FastAPI(title="Mosquito inspection scenario API", version=MODEL_VERSION)
app.add_middleware(CORSMiddleware,
    allow_origins=os.getenv("SIMULATION_CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,aedify-web.vercel.app").split(","),
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/")
def health():
    return {"status": "ok", "service": "Mosquito ABM API", "version": MODEL_VERSION}


@app.get("/references")
def references():
    return {"references": REFERENCES, "limitations": LIMITATIONS}


@app.post("/simulate")
def simulate(req: SimRequest):
    unique = {}
    for report in req.reports:
        if report.id in unique and unique[report.id] != report:
            raise HTTPException(422, "Conflicting duplicate report IDs")
        unique[report.id] = report
    selected = sorted((r for r in unique.values() if r.status not in CLOSED and not r.resolved
                       and r.resolved_at is None and (req.scope != "high_risk" or r.status in {"CRITICAL", "HIGH"})), key=lambda r: r.id)
    base = {"day": req.days, "model_version": MODEL_VERSION, "seed": req.seed, "scope": req.scope,
            "input_report_count": len(selected), "excluded_report_count": len(req.reports) - len(selected),
            "limitations": LIMITATIONS.copy(), "references": REFERENCES, "timeline": [],
            "weather_source": "Open-Meteo forecast", "generated_at": datetime.now(timezone.utc).isoformat()}
    if any(r.status in {"OPEN", "PENDING", "VERIFIED", "UNVERIFIED", "IN PROGRESS", "ASSIGNED"} for r in selected):
        base["limitations"].append("Some reports have workflow statuses instead of risk levels; these receive an assumed moderate weight. Preserve risk separately from workflow status to improve inputs.")
    if not selected:
        return {**base, "weather": None, "weather_source": "Not requested: no eligible reports", "weather_sequence": [],
                "total_agents": 0, "immature_agents": 0, "hotspot_predictions": [], "risk_index": 0,
                "summary": "No eligible unresolved reports. No sites were simulated; this does not establish absence of mosquitoes."}
    lat = sum(r.lat for r in selected) / len(selected)
    lng = sum(r.lng for r in selected) / len(selected)
    if any(distance_m(lat, lng, r.lat, r.lng) > 25000 for r in selected):
        raise HTTPException(422, "Select reports within a local 25 km area for a shared weather forecast.")
    if req.weather_override is not None:
        weather = [req.weather_override.model_copy(update={"date": (datetime.now(LOCAL_TZ).date() + timedelta(days=i)).isoformat()}) for i in range(1, req.days + 1)]
        base["weather_source"] = "User override (hypothetical constant weather)"
    else:
        weather = fetch_weather(lat, lng, req.days)
        base["weather_source"] = weather[0].provider + " forecast" + (" (cached)" if weather[0].cached else "")
        base["weather_fetched_at"] = weather[0].fetched_at
        base["weather_cached"] = weather[0].cached
    model = MosquitoModel(selected, req.seed)
    for day, w in enumerate(weather, 1):
        deaths, births = model.step(w)
        hotspots = predictions(model, w)
        base["timeline"].append({"day": day, "date": w.date, "total_agents": len(model.agents),
                                 "immature_agents": len(model.immatures), "deaths": deaths, "emerged": births,
                                 "agent_positions": [{"id": a.id, "lat": round(a.lat, 6), "lng": round(a.lng, 6),
                                                      "age_days": a.age, "source_report_id": a.source.id}
                                                     for a in model.agents[:100]],
                                 "hotspot_predictions": hotspots})
    if model.capped:
        base["limitations"].append("Agent capacity reached; population growth was truncated.")
    final = base["timeline"][-1]["hotspot_predictions"]
    return {**base, "weather": weather[-1].model_dump(), "weather_sequence": [w.model_dump() for w in weather],
            "weather_location": {"lat": lat, "lng": lng}, "forecast_start": weather[0].date,
            "total_agents": len(model.agents), "immature_agents": len(model.immatures),
            "hotspot_predictions": final, "risk_index": max((p["risk_score"] for p in final), default=0),
            "summary": f"Day {req.days} ({weather[-1].date}): {len(model.agents)} adult scenario units and {len(model.immatures)} immature units from {len(selected)} unresolved sites. "
                       + (f"Prioritize inspection {final[0]['location_estimate']}. " if final else "No adult clusters in this scenario; unresolved sites still need inspection. ")
                       + "One step equals one day. Relative inspection priority, not a dengue forecast."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
