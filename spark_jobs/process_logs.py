
import os
import sys
import re

# --------------------------------------------------
# 1. Configure Python for Spark on Windows
# --------------------------------------------------

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

# --------------------------------------------------
# 2. Import PySpark
# --------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType
)
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    substring,
    to_date,
    to_timestamp
)

# --------------------------------------------------
# 3. Define file paths
# --------------------------------------------------

INPUT_FILE = "dataset/raw/NASA_access_log_Jul95.gz"
OUTPUT_FOLDER = "dataset/processed"

# --------------------------------------------------
# 4. Define NASA log pattern
# --------------------------------------------------

LOG_PATTERN = re.compile(
    r'^(\S+) \S+ \S+ \[([^\]]+)\] "([^"]*)" (\d{3}) (\S+)'
)


# --------------------------------------------------
# 5. Function to parse each log line
# --------------------------------------------------

def parse_log(line):

    match = LOG_PATTERN.match(line)

    if not match:
        return None

    ip = match.group(1)
    timestamp = match.group(2)
    request = match.group(3)
    status = int(match.group(4))
    size = match.group(5)

    # Convert response size to integer
    if size == "-":
        bytes_sent = 0
    else:
        try:
            bytes_sent = int(size)
        except ValueError:
            bytes_sent = 0

    # Extract HTTP method, URL and protocol
    request_parts = request.split(" ", 2)

    if len(request_parts) == 3:
        method = request_parts[0]
        url = request_parts[1]
        protocol = request_parts[2]
    else:
        method = "UNKNOWN"
        url = request
        protocol = "UNKNOWN"

    return (
        ip,
        timestamp,
        method,
        url,
        protocol,
        status,
        bytes_sent
    )


# --------------------------------------------------
# 6. Create Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("NASA_Web_Usage_Mining")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n======================================")
print(" NASA WEB USAGE MINING PROJECT")
print("======================================")

try:

    # --------------------------------------------------
    # 7. Read NASA compressed log file
    # --------------------------------------------------

    print("\nSTEP 1: Reading NASA log dataset...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    logs_rdd = spark.sparkContext.textFile(INPUT_FILE)

    print("Dataset loaded successfully!")

    # --------------------------------------------------
    # 8. Parse log records using Spark RDD
    # --------------------------------------------------

    print("\nSTEP 2: Parsing web server logs...")

    parsed_rdd = logs_rdd.map(parse_log).cache()

    valid_rdd = parsed_rdd.filter(lambda x: x is not None).cache()

    total_lines = logs_rdd.count()
    invalid_lines = parsed_rdd.filter(lambda x: x is None).count()
    valid_records = valid_rdd.count()

    print("Total log lines:", total_lines)
    print("Valid records:", valid_records)
    print("Invalid records:", invalid_lines)

    # --------------------------------------------------
    # 9. Create Spark DataFrame
    # --------------------------------------------------

    print("\nSTEP 3: Creating Spark DataFrame...")

    schema = StructType([
        StructField("ip", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("method", StringType(), True),
        StructField("url", StringType(), True),
        StructField("protocol", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("bytes_sent", LongType(), True)
    ])

    df = spark.createDataFrame(valid_rdd, schema)

    print("\nSample records:")
    df.show(10, truncate=False)

    # --------------------------------------------------
    # 10. Calculate website statistics
    # --------------------------------------------------

    print("\nSTEP 4: Calculating website statistics...")

    total_requests = df.count()

    unique_visitors = df.select(
        countDistinct("ip")
    ).first()[0]

    total_data_bytes = df.agg(
        {"bytes_sent": "sum"}
    ).first()[0] or 0

    print("\nTotal HTTP requests:", total_requests)
    print("Unique visitors:", unique_visitors)
    print("Total response bytes:", total_data_bytes)

    # --------------------------------------------------
    # 11. Find most visited pages
    # --------------------------------------------------

    print("\nSTEP 5: Most visited pages...")

    popular_pages = (
        df.groupBy("url")
        .agg(count("*").alias("visits"))
        .orderBy(col("visits").desc())
    )

    popular_pages.show(10, truncate=False)

    # --------------------------------------------------
    # 12. Analyze HTTP status codes
    # --------------------------------------------------

    print("\nSTEP 6: HTTP status code analysis...")

    status_summary = (
        df.groupBy("status")
        .agg(count("*").alias("total"))
        .orderBy("status")
    )

    status_summary.show(truncate=False)

    # --------------------------------------------------
    # 13. Count failed requests
    # --------------------------------------------------

    print("\nSTEP 7: Failed request analysis...")

    failed_requests = df.filter(
        col("status") >= 400
    ).count()

    server_errors = df.filter(
        col("status") >= 500
    ).count()

    print("Failed requests (4xx and 5xx):", failed_requests)
    print("Server errors (5xx):", server_errors)

    # --------------------------------------------------
    # 14. Analyze daily website traffic
    # --------------------------------------------------

    print("\nSTEP 8: Daily website traffic...")

    daily_traffic = (
        df.withColumn(
            "request_date",
            to_date(
                substring(col("timestamp"), 1, 11),
                "dd/MMM/yyyy"
            )
        )
        .groupBy("request_date")
        .agg(count("*").alias("total_requests"))
        .orderBy("request_date")
    )

    daily_traffic.show(35, truncate=False)

    # --------------------------------------------------
    # 15. Save processed data
    # --------------------------------------------------

    print("\nSTEP 9: Saving processed results...")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df.write.mode("overwrite").parquet(
        f"{OUTPUT_FOLDER}/parsed_logs"
    )

    popular_pages.coalesce(1).write.mode("overwrite").option(
        "header", "true"
    ).csv(f"{OUTPUT_FOLDER}/popular_pages")

    status_summary.coalesce(1).write.mode("overwrite").option(
        "header", "true"
    ).csv(f"{OUTPUT_FOLDER}/status_summary")

    daily_traffic.coalesce(1).write.mode("overwrite").option(
        "header", "true"
    ).csv(f"{OUTPUT_FOLDER}/daily_traffic")

    print("\nProcessed data saved successfully!")

    # --------------------------------------------------
    # 16. Final summary
    # --------------------------------------------------

    print("\n======================================")
    print(" PROJECT PROCESSING SUMMARY")
    print("======================================")

    print("Total log lines:", total_lines)
    print("Valid records:", valid_records)
    print("Invalid records:", invalid_lines)
    print("Total requests:", total_requests)
    print("Unique visitors:", unique_visitors)
    print("Failed requests:", failed_requests)
    print("Server errors:", server_errors)

    print("\nSPARK PROCESSING COMPLETED SUCCESSFULLY!")

finally:

    # --------------------------------------------------
    # 17. Stop Spark session
    # --------------------------------------------------

    spark.stop()
    print("\nSpark session stopped.")
