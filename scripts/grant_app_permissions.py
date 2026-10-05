"""Post-deploy: give the app's auto-created service principal what it needs.

The bundle already grants CAN_QUERY on the gateway endpoint and SELECT on the index (app
resources). This script adds the rest from the identity model: USE CATALOG / USE SCHEMA /
SELECT on the chunks table, and CAN_USE on the Vector Search endpoint. It is idempotent.

    python scripts/grant_app_permissions.py --app policypilot-dev --catalog policypilot_dev
"""

from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, help="deployed app name, e.g. policypilot-dev")
    p.add_argument("--catalog", default="policypilot_dev")
    p.add_argument("--schema", default="filings")
    p.add_argument("--vs-endpoint", default="policypilot-vs-endpoint")
    args = p.parse_args()

    w = WorkspaceClient()
    sp = w.apps.get(args.app).service_principal_client_id
    if not sp:
        raise SystemExit(f"App {args.app} has no service principal yet; deploy it first.")

    schema = f"{args.catalog}.{args.schema}"
    grants = [
        ("catalog", args.catalog, ["USE_CATALOG"]),
        ("schema", schema, ["USE_SCHEMA"]),
        ("table", f"{schema}.chunks", ["SELECT"]),
        ("table", f"{schema}.chunks_index", ["SELECT"]),
    ]
    for securable_type, full_name, privileges in grants:
        w.api_client.do(
            "PATCH",
            f"/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}",
            body={"changes": [{"principal": sp, "add": privileges}]},
        )
        print(f"granted {privileges} on {securable_type} {full_name} to {sp}")

    endpoint = w.api_client.do("GET", f"/api/2.0/vector-search/endpoints/{args.vs_endpoint}")
    w.api_client.do(
        "PATCH",
        f"/api/2.0/permissions/vector-search-endpoints/{endpoint['id']}",
        body={"access_control_list": [{"service_principal_name": sp, "permission_level": "CAN_USE"}]},
    )
    print(f"granted CAN_USE on vector search endpoint {args.vs_endpoint} to {sp}")


if __name__ == "__main__":
    main()
