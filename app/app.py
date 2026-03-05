import os
from flask import Flask, jsonify, send_from_directory
from databricks.sdk import WorkspaceClient

app = Flask(__name__, static_folder="static")

# Load .env for local dev
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
CATALOG = "tim-kreutzfeldt-test"
SCHEMA = "adsb"
TABLE = "adsb_gold"

# DMV bounding box (covers IAD, DCA, and BWI with margin)
DC_BBOX = {"lat_min": 38.4, "lat_max": 39.8, "lon_min": -77.8, "lon_max": -76.4}


def _query(sql):
    w = WorkspaceClient()
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        format=None,
        wait_timeout="50s",
    )
    if not resp.result or not resp.manifest:
        return []
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array or []
    if resp.manifest.total_chunk_count and resp.manifest.total_chunk_count > 1:
        for i in range(1, resp.manifest.total_chunk_count):
            chunk = w.statement_execution.get_statement_result_chunk_n(
                resp.statement_id, i
            )
            rows.extend(chunk.data_array or [])
    return [dict(zip(cols, r)) for r in rows]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/data")
def get_data():
    sql = f"""
    SELECT icao24, lat, lon, heading, callsign, baroaltitude, velocity, vertrate,
           timestamp_bucket, maneuver_prediction, destination_prediction
    FROM `{CATALOG}`.`{SCHEMA}`.`{TABLE}`
    WHERE lat IS NOT NULL AND lon IS NOT NULL AND heading IS NOT NULL
      AND lat BETWEEN {DC_BBOX['lat_min']} AND {DC_BBOX['lat_max']}
      AND lon BETWEEN {DC_BBOX['lon_min']} AND {DC_BBOX['lon_max']}
    ORDER BY timestamp_bucket
    """
    data = _query(sql)
    return jsonify(data)
