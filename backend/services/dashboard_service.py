"""
Prism Dashboard Service — Creates Dashboards in Grafana, Datadog, Amplitude.

Provides async methods to create dashboards via platform APIs using
the DashboardSuggestion schema (which contains JSON strings for
queries, panels, and alerts).
"""

import json
from typing import Any

from observability.logging import get_logger
from utils.config import settings
from utils.connections import get_httpx_client

logger = get_logger(__name__)

class DashboardService:
    """Creates dashboards in Grafana, Datadog, and Amplitude."""

    # ── Grafana ───────────────────────────────────────────────────────────

    async def create_grafana_dashboard(self, suggestion: dict[str, Any]) -> dict:
        """
        Create a Grafana dashboard via the Grafana API.

        Args:
            suggestion: DashboardSuggestion dict with name, queries, panels, alerts fields.

        Returns:
            API response dict or error dict.
        """
        grafana_url = settings.grafana_url
        grafana_token = settings.grafana_service_account_token

        if not grafana_url or not grafana_token:
            logger.error("grafana_not_configured")
            return {"error": "Grafana URL or service account token not configured"}

        name = suggestion.get("name", "Untitled Dashboard")

        # Parse the JSON string fields
        try:
            queries = json.loads(suggestion.get("queries", "[]"))
            panels = json.loads(suggestion.get("panels", "[]"))
        except json.JSONDecodeError as e:
            logger.error("grafana_json_parse_error", error=str(e))
            return {"error": f"Invalid JSON in suggestion: {e}"}

        # Resolve query references in panel targets
        # Prism pattern: panels have "targets": ["A", "B"] which reference queries by refId
        for panel in panels:
            targets = panel.get("targets", [])
            if isinstance(targets, list):
                resolved_targets = []
                for target in targets:
                    if isinstance(target, str):
                        # Find the query with matching refId
                        for query in queries:
                            if query.get("refId") == target:
                                resolved_targets.append(query)
                                break
                        else:
                            resolved_targets.append(target)
                    else:
                        resolved_targets.append(target)
                panel["targets"] = resolved_targets

        # Build Grafana dashboard JSON payload
        dashboard_payload = {
            "dashboard": {
                "id": None,
                "title": name,
                "tags": ["auto-generated", "prism", "observability"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 1,
                "refresh": "5s",
                "panels": panels,
            },
            "folderId": 0,
            "overwrite": True,
        }

        url = f"{grafana_url.rstrip('/')}/api/dashboards/db"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {grafana_token}",
        }

        client = get_httpx_client()
        try:
            response = await client.post(url, json=dashboard_payload, headers=headers, timeout=30)
            if response.status_code < 200 or response.status_code > 299:
                logger.error(
                    "grafana_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                return {"error": f"Grafana API error ({response.status_code}): {response.text[:200]}"}
            logger.info("grafana_dashboard_created", name=name)
            return response.json()
        except Exception as e:
            logger.error("grafana_request_error", error=str(e))
            return {"error": str(e)}

    # ── Datadog ───────────────────────────────────────────────────────────

    async def create_datadog_dashboard(self, suggestion: dict[str, Any]) -> dict:
        """
        Create a Datadog dashboard via the Datadog API.

        Uses httpx instead of the Datadog Go SDK — builds the widget JSON directly.
        """
        dd_api_key = settings.datadog_api_key
        dd_app_key = settings.datadog_app_key

        if not dd_api_key or not dd_app_key:
            logger.error("datadog_not_configured")
            return {"error": "Datadog API key or App key not configured"}

        name = suggestion.get("name", "Untitled Dashboard")

        try:
            queries = json.loads(suggestion.get("queries", "[]"))
            panels = json.loads(suggestion.get("panels", "[]"))
        except json.JSONDecodeError as e:
            logger.error("datadog_json_parse_error", error=str(e))
            return {"error": f"Invalid JSON in suggestion: {e}"}

        # Build Datadog widgets from panels
        widgets = []
        for i, panel in enumerate(panels):
            title = panel.get("title", f"Panel {i + 1}")
            grid_pos = panel.get("gridPos", {"h": 4, "w": 6, "x": 0, "y": 0})

            # Resolve query for the panel
            query_str = "avg:system.cpu.user{*}"
            panel_targets = panel.get("targets", [])
            for target in panel_targets:
                if isinstance(target, str):
                    for q in queries:
                        if q.get("refId") == target:
                            query_str = q.get("expr") or q.get("query", query_str)
                            break
                elif isinstance(target, dict):
                    query_str = target.get("expr") or target.get("query", query_str)

            widget = {
                "definition": {
                    "type": "timeseries",
                    "title": title,
                    "requests": [
                        {
                            "q": query_str,
                            "display_type": "line",
                            "style": {"palette": "dog_classic", "line_type": "solid", "line_width": "normal"},
                        }
                    ],
                },
                "layout": {
                    "x": grid_pos.get("x", 0),
                    "y": grid_pos.get("y", i * 4),
                    "width": min(grid_pos.get("w", 6), 12),
                    "height": grid_pos.get("h", 4),
                },
            }
            widgets.append(widget)

        if not widgets:
            widgets = [
                {
                    "definition": {"type": "timeseries", "title": "Default", "requests": [{"q": "avg:system.cpu.user{*}"}]},
                    "layout": {"x": 0, "y": 0, "width": 12, "height": 4},
                }
            ]

        dashboard_body = {
            "title": name,
            "description": "Auto-generated by Prism AI Code Reviewer",
            "layout_type": "ordered",
            "widgets": widgets,
            "template_variables": [{"name": "env", "prefix": "env", "default": "*"}],
            "notify_list": [],
        }

        # Datadog API endpoint — configurable for different regions
        dd_host = settings.datadog_host or "api.datadoghq.com"
        url = f"https://{dd_host}/api/v1/dashboard"
        headers = {
            "Content-Type": "application/json",
            "DD-API-KEY": dd_api_key,
            "DD-APPLICATION-KEY": dd_app_key,
        }

        client = get_httpx_client()
        try:
            response = await client.post(url, json=dashboard_body, headers=headers, timeout=30)
            if response.status_code < 200 or response.status_code > 299:
                logger.error(
                    "datadog_dashboard_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                return {"error": f"Datadog API error ({response.status_code}): {response.text[:200]}"}
            logger.info("datadog_dashboard_created", name=name)
            return response.json()
        except Exception as e:
            logger.error("datadog_dashboard_request_error", error=str(e))
            return {"error": str(e)}

    # ── Amplitude ─────────────────────────────────────────────────────────

    async def create_amplitude_dashboard(self, suggestion: dict[str, Any]) -> dict:
        """
        Create an Amplitude dashboard.
        Note: Amplitude's Dashboard API is limited; this posts a placeholder.
        Full Amplitude chart creation requires the Amplitude Analytics API.
        """
        amplitude_api_key = settings.amplitude_api_key
        amplitude_secret_key = settings.amplitude_secret_key

        if not amplitude_api_key or not amplitude_secret_key:
            logger.warning("amplitude_not_configured")
            return {"error": "Amplitude API key or secret not configured", "status": "skipped"}

        name = suggestion.get("name", "Untitled Dashboard")
        logger.info("amplitude_dashboard_placeholder", name=name)

        # Amplitude doesn't have a straightforward dashboard creation API
        # like Grafana/Datadog. Real implementation would use Amplitude's
        # Chart/Dashboard REST API or create event segmentation charts.
        return {
            "status": "placeholder",
            "name": name,
            "message": "Amplitude dashboard creation requires manual setup. "
                       "Event names and properties have been identified for configuration.",
        }

    # ── Router ────────────────────────────────────────────────────────────

    async def create_dashboard(self, suggestion: dict[str, Any]) -> dict:
        """Route a dashboard suggestion to the correct platform."""
        dashboard_type = suggestion.get("type", "grafana").lower()

        if dashboard_type == "grafana":
            return await self.create_grafana_dashboard(suggestion)
        elif dashboard_type == "datadog":
            return await self.create_datadog_dashboard(suggestion)
        elif dashboard_type == "amplitude":
            return await self.create_amplitude_dashboard(suggestion)
        else:
            logger.warning("unknown_dashboard_type", type=dashboard_type)
            return {"error": f"Unknown dashboard type: {dashboard_type}"}
