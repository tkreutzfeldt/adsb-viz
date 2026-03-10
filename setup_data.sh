#!/usr/bin/env bash
# Extracts raw data and uploads to Databricks UC Volume
set -euo pipefail

CATALOG="${1:-tim-kreutzfeldt-test}"
SCHEMA="${2:-adsb}"
PROFILE="${3:-fe-sandbox-serverless}"

VOLUME_PATH="/Volumes/${CATALOG}/${SCHEMA}/raw"

echo "Extracting data archive..."
mkdir -p data/raw
tar -xzf data/adsb_data.tar.gz -C data/raw

echo "Creating schema and volume..."
databricks sql exec --profile "$PROFILE" \
  --statement "CREATE SCHEMA IF NOT EXISTS \`${CATALOG}\`.\`${SCHEMA}\`"
databricks sql exec --profile "$PROFILE" \
  --statement "CREATE VOLUME IF NOT EXISTS \`${CATALOG}\`.\`${SCHEMA}\`.raw"

echo "Uploading adsb_bronze.parquet to ${VOLUME_PATH}..."
databricks fs cp data/raw/adsb_bronze.parquet "dbfs:${VOLUME_PATH}/adsb_bronze.parquet" \
  --profile "$PROFILE" --overwrite

echo "Done! Data uploaded to ${VOLUME_PATH}"
