# Databricks notebook source
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "tim-kreutzfeldt-test", "Catalog")
dbutils.widgets.text("schema", "adsb", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Create DC Metro Silver Table
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define Washington DC metro area bounding box (4x original size)
# This encompasses IAD, DCA, BWI airports and surrounding area
DC_LAT_MIN = 35.75
DC_LAT_MAX = 41.75
DC_LON_MIN = -81.0
DC_LON_MAX = -73.0

# Read bronze table and filter for DC metro area
df_bronze = spark.table(f"`{catalog}`.`{schema}`.adsb_bronze")

df_filtered = df_bronze.filter(
    (df_bronze.lat >= DC_LAT_MIN) & 
    (df_bronze.lat <= DC_LAT_MAX) & 
    (df_bronze.lon >= DC_LON_MIN) & 
    (df_bronze.lon <= DC_LON_MAX) &
    # Filter out rows with nulls in critical columns
    df_bronze.lat.isNotNull() &
    df_bronze.lon.isNotNull() &
    df_bronze.heading.isNotNull() &
    df_bronze.velocity.isNotNull() &
    df_bronze.vertrate.isNotNull() &
    df_bronze.callsign.isNotNull() &
    df_bronze.baroaltitude.isNotNull() &
    df_bronze.geoaltitude.isNotNull()
)

# Create timestamp column from Unix timestamp
df_with_timestamp = df_filtered.withColumn(
    "timestamp",
    F.from_unixtime(F.col("time")).cast("timestamp")
)

# Create timestamp_bucket - truncate to 10-second intervals
# Floor the Unix timestamp to nearest 10-second bucket
df_with_bucket = df_with_timestamp.withColumn(
    "timestamp_bucket",
    F.from_unixtime(F.floor(F.col("time") / 10) * 10).cast("timestamp")
)

# For each icao24 and bucket, keep only the last row (max timestamp)
# Define window partitioned by icao24 and bucket, ordered by timestamp
window_spec = Window.partitionBy("icao24", "timestamp_bucket").orderBy(F.desc("timestamp"))

df_with_row_num = df_with_bucket.withColumn(
    "row_num",
    F.row_number().over(window_spec)
)

# Keep only the last row per bucket (row_num = 1)
df_silver = df_with_row_num.filter(F.col("row_num") == 1).drop("row_num")

# Show summary
original_count = df_filtered.count()
final_count = df_silver.count()
print(f"Records in DC metro area (after null filtering): {original_count:,}")
print(f"After 10-second bucketing per icao24: {final_count:,}")
print(f"Reduction from bucketing: {original_count - final_count:,} records ({(1 - final_count/original_count)*100:.1f}%)")

# Write to silver table with schema overwrite
df_silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"`{catalog}`.`{schema}`.adsb_silver")

print("\n✓ Silver table created: tim-kreutzfeldt-test.adsb.adsb_silver")

# Display sample
display(df_silver.select("timestamp", "timestamp_bucket", "icao24", "callsign", "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude").orderBy("timestamp").limit(10))

# COMMAND ----------

# DBTITLE 1,Add Acceleration Field
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read the current silver table
df = spark.table(f"`{catalog}`.`{schema}`.adsb_silver")

# Define window for calculating lag values, partitioned by icao24 and ordered by timestamp
window_spec = Window.partitionBy("icao24").orderBy("timestamp")

# Calculate acceleration as derivative of velocity over time
df_with_accel = df.withColumn(
    "prev_velocity", F.lag("velocity").over(window_spec)
).withColumn(
    "prev_timestamp", F.lag("timestamp").over(window_spec)
).withColumn(
    # Calculate time difference in seconds
    "time_diff", 
    (F.unix_timestamp("timestamp") - F.unix_timestamp("prev_timestamp"))
).withColumn(
    # Calculate acceleration (m/s²)
    "acceleration_raw",
    F.when(
        F.col("time_diff") > 0,
        (F.col("velocity") - F.col("prev_velocity")) / F.col("time_diff")
    ).otherwise(0)
)

# Fill first row of each aircraft with the second row's value
window_fill = Window.partitionBy("icao24").orderBy("timestamp").rowsBetween(0, 1)
df_final = df_with_accel.withColumn(
    "acceleration",
    F.when(F.col("prev_velocity").isNull(), 
           F.first(F.col("acceleration_raw"), ignorenulls=True).over(window_fill)
    ).otherwise(F.col("acceleration_raw"))
).drop("prev_velocity", "prev_timestamp", "time_diff", "acceleration_raw")

# Overwrite the silver table with the new acceleration column
df_final.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"`{catalog}`.`{schema}`.adsb_silver")

print("✓ Added acceleration field to adsb_silver table")
display(df_final.select("timestamp", "icao24", "callsign", "velocity", "acceleration", "lat", "lon").orderBy("icao24", "timestamp").limit(20))