#!/bin/bash
# Sync and deploy the ADSB viz app to Databricks
PROFILE="fe-sandbox-serverless"
APP_NAME="adsb-viz"
SOURCE="/Users/tim.kreutzfeldt/workspaces/adsb_dmv/app"
TARGET="/Workspace/Users/tim.kreutzfeldt@databricks.com/adsb-viz"

echo "Syncing files..."
databricks --profile "$PROFILE" sync "$SOURCE" "$TARGET"

echo "Deploying..."
databricks --profile "$PROFILE" apps deploy "$APP_NAME" \
  --source-code-path "$TARGET"

echo "Done! App URL: https://adsb-viz-7474647294135428.aws.databricksapps.com"
