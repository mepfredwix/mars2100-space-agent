"""Allowlisted space tools, schemas, argument validation, and execution."""

import json
import re
from typing import Any

import requests


class ToolValidationError(ValueError):
    """Raised internally when a model supplies invalid tool arguments."""


def get_mars_info() -> dict[str, Any]:
    """Return basic, static Mars facts (not live telemetry)."""
    return {
        "name": "Mars",
        "position_from_sun": 4,
        "diameter_km": 6779,
        "average_distance_from_sun_million_km": 227.9,
        "gravity_m_per_s2": 3.71,
        "day_length": "24 hours 37 minutes",
        "year_length_earth_days": 687,
        "moons": ["Phobos", "Deimos"],
        "atmosphere": "Mostly carbon dioxide, with a thin surface atmosphere",
        "note": "Built-in reference facts; values are approximate and not live data.",
    }


def get_space_mission_info(mission_name: str) -> dict[str, Any]:
    """Return basic static information for a supported space mission."""
    missions = {
        "apollo 11": {
            "name": "Apollo 11",
            "agency": "NASA",
            "launch_date": "1969-07-16",
            "type": "Crewed lunar landing",
            "summary": (
                "Neil Armstrong and Buzz Aldrin landed on the Moon while "
                "Michael Collins orbited in the command module."
            ),
        },
        "voyager 1": {
            "name": "Voyager 1",
            "agency": "NASA",
            "launch_date": "1977-09-05",
            "type": "Robotic outer-planet and interstellar probe",
            "summary": (
                "Voyager 1 explored Jupiter and Saturn before continuing "
                "on its extended interstellar mission."
            ),
        },
        "artemis i": {
            "name": "Artemis I",
            "agency": "NASA",
            "launch_date": "2022-11-16",
            "type": "Uncrewed lunar flight test",
            "summary": (
                "The Space Launch System sent Orion around the Moon and "
                "back to Earth to test the integrated system."
            ),
        },
        "perseverance": {
            "name": "Perseverance",
            "agency": "NASA",
            "launch_date": "2020-07-30",
            "type": "Mars rover mission",
            "summary": (
                "The rover landed in Jezero Crater on 2021-02-18 to study "
                "Mars, search for signs of ancient microbial life, and cache samples."
            ),
        },
    }

    mission = missions.get(mission_name.strip().casefold())
    if mission is None:
        return {
            "found": False,
            "requested_mission": mission_name,
            "message": "This mission is not in the small built-in dataset.",
            "available_missions": [item["name"] for item in missions.values()],
        }

    return {"found": True, **mission, "note": "Built-in reference data, not live status."}


def _short_description(description: str) -> str:
    """Normalize whitespace and keep at most two compact sentences."""
    normalized = " ".join(description.split())
    if not normalized:
        return "No description provided."

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    shortened = " ".join(sentences[:2]).strip()
    if len(shortened) > 500:
        shortened = shortened[:497].rsplit(" ", 1)[0] + "..."
    return shortened


def search_space_data(query: str) -> dict[str, Any]:
    """Search NASA's Image and Video Library for up to five image results."""
    try:
        response = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": "image"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return {
            "error": "nasa_api_timeout",
            "message": "The NASA image search timed out. Please try again later.",
        }
    except requests.RequestException as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        return {
            "error": "nasa_api_http_error",
            "status_code": status_code,
            "message": "The NASA image search could not be completed.",
        }
    except ValueError:
        return {
            "error": "nasa_api_invalid_response",
            "message": "NASA returned a response that was not valid JSON.",
        }

    items = payload.get("collection", {}).get("items", [])
    if not isinstance(items, list):
        return {
            "error": "nasa_api_unexpected_response",
            "message": "NASA returned an unexpected result format.",
        }

    results = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("data"), list):
            continue
        metadata = item["data"][0] if item["data"] else {}
        if not isinstance(metadata, dict):
            continue

        results.append(
            {
                "title": metadata.get("title") or "Untitled",
                "date_created": metadata.get("date_created") or "Unknown",
                "description": _short_description(
                    str(metadata.get("description") or "")
                ),
            }
        )
        if len(results) == 5:
            break

    if not results:
        return {
            "query": query,
            "results": [],
            "message": "No NASA image results were found for this query.",
        }

    return {"query": query, "result_count": len(results), "results": results}


# These OpenAI-compatible JSON schemas tell the model what tools exist and
# which arguments are allowed. No free-form executable code is accepted.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_mars_info",
            "description": "Get basic reference facts about the planet Mars.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_space_mission_info",
            "description": (
                "Get basic reference information about a space mission. "
                "Built-in examples include Apollo 11, Voyager 1, Artemis I, "
                "and Perseverance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mission_name": {
                        "type": "string",
                        "description": "Name of the mission, for example Apollo 11.",
                    }
                },
                "required": ["mission_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_space_data",
            "description": (
                "Search the NASA Image and Video Library for images. "
                "Returns up to five results with titles, dates, and short descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms, for example Mars rovers.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

# Explicit dispatch is intentionally separate from the model output. Never use
# eval(), exec(), or model-generated import/code execution here.
TOOL_HANDLERS = {
    "get_mars_info": get_mars_info,
    "get_space_mission_info": get_space_mission_info,
    "search_space_data": search_space_data,
}


def _validate_arguments(
    tool_name: str, arguments: dict[str, Any]
) -> None:
    """Apply per-tool allowlists rather than trusting model-generated JSON."""
    required: set[str] = set()
    allowed: set[str] = set()

    string_argument = {
        "get_space_mission_info": "mission_name",
        "search_space_data": "query",
    }.get(tool_name)
    if string_argument is not None:
        required = {string_argument}
        allowed = {string_argument}

    missing = sorted(required - arguments.keys())
    unexpected = sorted(arguments.keys() - allowed)
    if missing:
        raise ToolValidationError(f"Missing required argument(s): {', '.join(missing)}")
    if unexpected:
        raise ToolValidationError(
            f"Unexpected argument(s): {', '.join(unexpected)}"
        )

    if string_argument is not None:
        value = arguments[string_argument]
        if not isinstance(value, str) or not value.strip():
            raise ToolValidationError(
                f"{string_argument} must be a non-empty string"
            )


def execute_tool(tool_name: str, arguments_json: str) -> str:
    """Validate and execute one allowlisted tool, returning JSON for the LLM.

    Errors are returned as structured observations instead of crashing the
    entire agent, so the model gets an opportunity to correct its next step.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps(
            {
                "error": "unknown_tool",
                "tool": tool_name,
                "available_tools": sorted(TOOL_HANDLERS),
            }
        )

    try:
        arguments = json.loads(arguments_json)
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be a JSON object.")
        _validate_arguments(tool_name, arguments)
    except json.JSONDecodeError:
        return json.dumps(
            {
                "error": "invalid_json",
                "tool": tool_name,
                "message": "Arguments must be valid JSON.",
            }
        )
    except ToolValidationError as exc:
        return json.dumps(
            {
                "error": "invalid_arguments",
                "tool": tool_name,
                "message": str(exc),
            }
        )

    try:
        result = handler(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        # Do not expose internal exception details or stack traces to the model.
        return json.dumps(
            {
                "error": "tool_execution_failed",
                "tool": tool_name,
                "message": "The tool could not complete its built-in operation.",
            }
        )
