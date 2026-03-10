# Databricks notebook source
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "tim-kreutzfeldt-test", "Catalog")
dbutils.widgets.text("schema", "adsb", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

VOLUME_PATH = f"/Volumes/{catalog}/{schema}/raw"
PARQUET_FILE = f"{VOLUME_PATH}/adsb_bronze.parquet"
BRONZE_TABLE = f"`{catalog}`.`{schema}`.adsb_bronze"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Source: {PARQUET_FILE}")
print(f"Target: {BRONZE_TABLE}")

# COMMAND ----------

# DBTITLE 1,Ensure schema and volume exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.raw")

print(f"Schema and volume ready in `{catalog}`.`{schema}`")

# COMMAND ----------

# DBTITLE 1,Load parquet from volume into bronze table
df_bronze = spark.read.parquet(PARQUET_FILE)

print(f"Loaded {df_bronze.count():,} rows from parquet")
print(f"Columns: {df_bronze.columns}")

df_bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BRONZE_TABLE)

print(f"Created bronze table: {BRONZE_TABLE}")
display(df_bronze.limit(5))
