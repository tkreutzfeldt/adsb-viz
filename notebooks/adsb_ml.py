# Databricks notebook source
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "tim-kreutzfeldt-test", "Catalog")
dbutils.widgets.text("schema", "adsb", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Load Data and Create Labels
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math

# Define airport locations (lat, lon)
airports = {
    'IAD': (38.9531, -77.4565),  # Dulles
    'DCA': (38.8521, -77.0377),  # Reagan National
    'BWI': (39.1774, -76.6684)   # Baltimore/Washington
}

# Define reasonable range in degrees (~20 km radius)
# At this latitude, 1 degree lat ≈ 111 km, 1 degree lon ≈ 85 km
RANGE_DEGREES = 0.18  # approximately 20 km

# Load silver table
df = spark.table(f"`{catalog}`.`{schema}`.adsb_silver")

# Convert baroaltitude from meters to feet
df = df.withColumn("altitude_ft", F.col("baroaltitude") * 3.28084)

# Calculate distance to each airport
for airport_name, (airport_lat, airport_lon) in airports.items():
    df = df.withColumn(
        f"dist_to_{airport_name}",
        F.sqrt(
            F.pow(F.col("lat") - F.lit(airport_lat), 2) + 
            F.pow(F.col("lon") - F.lit(airport_lon), 2)
        )
    )

# Check if near any airport
df = df.withColumn(
    "near_airport",
    (F.col("dist_to_IAD") < RANGE_DEGREES) |
    (F.col("dist_to_DCA") < RANGE_DEGREES) |
    (F.col("dist_to_BWI") < RANGE_DEGREES)
)

# Get first and last timestamp for each callsign
window_first = Window.partitionBy("callsign").orderBy("timestamp")
window_last = Window.partitionBy("callsign").orderBy(F.desc("timestamp"))

df = df.withColumn("row_num_first", F.row_number().over(window_first))
df = df.withColumn("row_num_last", F.row_number().over(window_last))

# Mark first and last data points
df = df.withColumn("is_first_point", F.col("row_num_first") == 1)
df = df.withColumn("is_last_point", F.col("row_num_last") == 1)

# For each callsign, get whether first/last points are near airport
first_near = df.filter(F.col("is_first_point")).select(
    "callsign", 
    F.col("near_airport").alias("first_near_airport")
)
last_near = df.filter(F.col("is_last_point")).select(
    "callsign", 
    F.col("near_airport").alias("last_near_airport")
)

# Join back to main dataframe
df = df.join(first_near, on="callsign", how="left")
df = df.join(last_near, on="callsign", how="left")

# Create labels with 3000 ft altitude threshold (2x increase)
df = df.withColumn(
    "label",
    F.when(
        (F.col("altitude_ft") < 3000) & (F.col("first_near_airport")),
        "takeoff"
    ).when(
        (F.col("altitude_ft") < 3000) & (F.col("last_near_airport")),
        "landing"
    ).otherwise("cruise")
)

print("Label distribution:")
df.groupBy("label").count().orderBy("label").show()

# Select features for modeling (excluding acceleration for now)
feature_cols = [
    "lat", "lon", "velocity", "heading", "vertrate", "altitude_ft", "label"
]

df_model = df.select(feature_cols).na.drop()

print(f"\nTotal records for modeling: {df_model.count():,}")
print("\nFeature summary:")
df_model.describe().show()

# Save to a table for AutoML
df_model.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.flight_phase_training")
print("\n✓ Training data saved to tim-kreutzfeldt-test.adsb.flight_phase_training")

# COMMAND ----------

# DBTITLE 1,Train Classifier Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Load the prepared training data
df_training = spark.table(f"`{catalog}`.`{schema}`.flight_phase_training")

print(f"Total records: {df_training.count():,}")
print("\nClass distribution:")
df_training.groupBy("label").count().orderBy("label").show()

# Convert to pandas for sklearn
print("\nConverting to pandas...")
pdf = df_training.toPandas()

# Prepare features and labels
feature_columns = ["lat", "lon", "velocity", "heading", "vertrate", "altitude_ft"]
X = pdf[feature_columns]
y = pdf["label"]

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split: 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\nData splits:")
print(f"  Train: {len(X_train):,} samples")
print(f"  Validation: {len(X_val):,} samples")
print(f"  Test: {len(X_test):,} samples")

# Set MLflow experiment
mlflow.set_experiment("/Users/tim.kreutzfeldt@databricks.com/flight_phase_classifier")

# Train multiple models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
}

results = []

print("\n" + "="*70)
print("Training Models...")
print("="*70)

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("val_accuracy", val_accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        results.append({
            "model": model_name,
            "val_f1_score": val_f1,
            "val_accuracy": val_accuracy
        })
        
        print(f"  Validation F1 Score: {val_f1:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")

# Find best model
results_df = pd.DataFrame(results).sort_values("val_f1_score", ascending=False)
print("\n" + "="*70)
print("Model Comparison (sorted by F1 Score):")
print("="*70)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

print(f"\n" + "="*70)
print(f"Best Model: {best_model_name}")
print("="*70)

# Evaluate best model on test set
y_test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTest Set Performance:")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(f"\nRows: True labels, Columns: Predicted labels")
print(f"Order: {sorted(y_test.unique())}")

print(f"\n✓ Training complete! Check MLflow UI for detailed results.")

# COMMAND ----------

# DBTITLE 1,Create Gold Table with Predictions
import mlflow
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Get the best model from MLflow experiment (most recent RandomForest run)
print("Finding best model from MLflow experiment...")
experiment_name = "/Users/tim.kreutzfeldt@databricks.com/flight_phase_classifier"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"ERROR: Experiment '{experiment_name}' not found. Please run the training cell first.")
else:
    # Get most recent runs, then filter for RandomForest (which is typically the best)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["start_time DESC"]
    )
    
    if len(runs) == 0:
        print("ERROR: No runs found in experiment. Please run the training cell first.")
    else:
        # Get the most recent RandomForest run
        rf_runs = runs[runs['params.model_type'] == 'RandomForest']
        
        if len(rf_runs) == 0:
            print("ERROR: No RandomForest runs found. Using most recent run.")
            best_run = runs.iloc[0]
        else:
            best_run = rf_runs.iloc[0]
        
        best_run_id = best_run['run_id']
        best_model_name = best_run['params.model_type']
        best_f1 = best_run['metrics.val_f1_score']
        best_start_time = best_run['start_time']
        
        print(f"✓ Found best model: {best_model_name}")
        print(f"  Run ID: {best_run_id}")
        print(f"  Start Time: {best_start_time}")
        print(f"  Validation F1: {best_f1:.4f}")
        
        # Load the best model
        print("\nLoading model...")
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.sklearn.load_model(model_uri)
        print("✓ Model loaded successfully")
        
        # Load silver table
        print("\nLoading silver table...")
        df_silver = spark.table(f"`{catalog}`.`{schema}`.adsb_silver")
        print(f"Total records: {df_silver.count():,}")
        
        # Prepare features for prediction
        df_silver = df_silver.withColumn("altitude_ft", F.col("baroaltitude") * 3.28084)
        
        # Convert to pandas for prediction
        print("\nConverting to pandas for prediction...")
        pdf = df_silver.toPandas()
        
        # Prepare feature matrix
        feature_columns = ["lat", "lon", "velocity", "heading", "vertrate", "altitude_ft"]
        X_pred = pdf[feature_columns]
        
        # Make predictions
        print("Making predictions...")
        pdf['maneuver_prediction'] = best_model.predict(X_pred)
        
        print("\nPrediction distribution:")
        print(pdf['maneuver_prediction'].value_counts())
        
        # Calculate destination for landing predictions
        print("\nCalculating destination predictions for landing maneuvers...")
        
        # Define airport locations
        airports = {
            'IAD': (38.9531, -77.4565),
            'DCA': (38.8521, -77.0377),
            'BWI': (39.1774, -76.6684)
        }
        
        def get_closest_airport(row):
            if row['maneuver_prediction'] != 'landing':
                return None
            
            lat, lon = row['lat'], row['lon']
            min_dist = float('inf')
            closest = None
            
            for airport_name, (airport_lat, airport_lon) in airports.items():
                dist = ((lat - airport_lat)**2 + (lon - airport_lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest = airport_name
            
            return closest
        
        pdf['destination_prediction'] = pdf.apply(get_closest_airport, axis=1)
        
        print("\nDestination prediction distribution:")
        print(pdf['destination_prediction'].value_counts(dropna=False))
        
        # Convert back to Spark DataFrame
        print("\nConverting back to Spark DataFrame...")
        df_gold = spark.createDataFrame(pdf)
        
        # Save as gold table
        print("\nSaving gold table...")
        df_gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"`{catalog}`.`{schema}`.adsb_gold")
        
        print("\n" + "="*70)
        print("✓ Gold table created: tim-kreutzfeldt-test.adsb.adsb_gold")
        print("="*70)
        
        # Display sample
        print("\nSample predictions:")
        display(df_gold.select(
            "timestamp", "icao24", "callsign", "lat", "lon", 
            "altitude_ft", "velocity", "maneuver_prediction", "destination_prediction"
        ).orderBy("timestamp").limit(20))