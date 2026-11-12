
import os
import sys

# --------------------------------------------------
# 1. Configure Python and Hadoop for Windows
# --------------------------------------------------

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

os.environ["HADOOP_HOME"] = r"C:\hadoop"

os.environ["PATH"] = (
    os.environ["HADOOP_HOME"] + r"\bin;"
    + os.environ.get("PATH", "")
)

# --------------------------------------------------
# 2. Import PySpark libraries
# --------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    sum as spark_sum,
    avg,
    when
)

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# --------------------------------------------------
# 3. Define project paths
# --------------------------------------------------

INPUT_PATH = "dataset/processed/parsed_logs"
MODEL_PATH = "models/visitor_kmeans"
OUTPUT_PATH = "dataset/processed"

# --------------------------------------------------
# 4. Create Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("NASA_Visitor_Clustering")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n======================================")
print(" NASA VISITOR MACHINE LEARNING")
print("======================================")

try:

    # --------------------------------------------------
    # 5. Load processed NASA records
    # --------------------------------------------------

    print("\nSTEP 1: Loading processed NASA data...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            "Processed dataset not found: " + INPUT_PATH
        )

    df = spark.read.parquet(INPUT_PATH)

    print("Processed records loaded:", df.count())

    # --------------------------------------------------
    # 6. Create visitor-level features
    # --------------------------------------------------

    print("\nSTEP 2: Extracting visitor features...")

    visitor_features = (
        df.groupBy("ip")
        .agg(
            count("*").alias("total_requests"),

            countDistinct("url").alias("unique_pages"),

            spark_sum("bytes_sent").alias("total_bytes"),

            spark_sum(
                when(col("status") >= 400, 1).otherwise(0)
            ).alias("failed_requests"),

            spark_sum(
                when(col("method") == "GET", 1).otherwise(0)
            ).alias("get_requests")
        )
        .fillna(0)
    )

    print("Unique visitors:", visitor_features.count())

    print("\nVisitor feature sample:")

    visitor_features.show(10, truncate=False)

    # --------------------------------------------------
    # 7. Convert features into a vector
    # --------------------------------------------------

    print("\nSTEP 3: Preparing machine learning features...")

    feature_columns = [
        "total_requests",
        "unique_pages",
        "total_bytes",
        "failed_requests",
        "get_requests"
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="raw_features"
    )

    assembled_data = assembler.transform(visitor_features)

    # --------------------------------------------------
    # 8. Standardize the features
    # --------------------------------------------------

    print("\nSTEP 4: Scaling numerical features...")

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    scaler_model = scaler.fit(assembled_data)

    scaled_data = scaler_model.transform(assembled_data)

    # --------------------------------------------------
    # 9. Train K-Means clustering model
    # --------------------------------------------------

    print("\nSTEP 5: Training K-Means model...")

    kmeans = KMeans(
        k=3,
        seed=42,
        featuresCol="features",
        predictionCol="cluster"
    )

    kmeans_model = kmeans.fit(scaled_data)

    print("K-Means model trained successfully!")

    # --------------------------------------------------
    # 10. Predict visitor clusters
    # --------------------------------------------------

    print("\nSTEP 6: Predicting visitor clusters...")

    predictions = kmeans_model.transform(scaled_data)

    print("\nCluster prediction sample:")

    predictions.select(
        "ip",
        "total_requests",
        "unique_pages",
        "total_bytes",
        "failed_requests",
        "cluster"
    ).show(15, truncate=False)

    # --------------------------------------------------
    # 11. Evaluate the clustering model
    # --------------------------------------------------

    print("\nSTEP 7: Evaluating clustering model...")

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    silhouette = evaluator.evaluate(predictions)

    print("Silhouette score:", silhouette)

    # --------------------------------------------------
    # 12. Count visitors in each cluster
    # --------------------------------------------------

    print("\nSTEP 8: Analyzing visitor clusters...")

    cluster_summary = (
        predictions.groupBy("cluster")
        .agg(
            count("*").alias("visitor_count"),
            avg("total_requests").alias("avg_requests"),
            avg("unique_pages").alias("avg_unique_pages"),
            avg("total_bytes").alias("avg_bytes"),
            avg("failed_requests").alias("avg_failed_requests")
        )
        .orderBy("cluster")
    )

    cluster_summary.show(truncate=False)

    # --------------------------------------------------
    # 13. Save trained model
    # --------------------------------------------------

    print("\nSTEP 9: Saving trained model...")

    os.makedirs("models", exist_ok=True)

    # Save both the scaler and K-Means model
    scaler_model.write().overwrite().save(
        "models/visitor_scaler"
    )

    kmeans_model.write().overwrite().save(
        MODEL_PATH
    )

    print("Scaler saved successfully!")
    print("K-Means model saved successfully!")

    # --------------------------------------------------
    # 14. Save visitor predictions
    # --------------------------------------------------

    print("\nSTEP 10: Saving visitor predictions...")

    predictions.select(
        "ip",
        "total_requests",
        "unique_pages",
        "total_bytes",
        "failed_requests",
        "get_requests",
        "cluster"
    ).write.mode("overwrite").parquet(
        f"{OUTPUT_PATH}/visitor_clusters"
    )

    cluster_summary.coalesce(1).write.mode(
        "overwrite"
    ).option(
        "header", "true"
    ).csv(
        f"{OUTPUT_PATH}/cluster_summary"
    )

    print("Visitor predictions saved!")
    print("Cluster summary saved!")

    # --------------------------------------------------
    # 15. Final project summary
    # --------------------------------------------------

    print("\n======================================")
    print(" MACHINE LEARNING SUMMARY")
    print("======================================")

    print("Algorithm: K-Means Clustering")
    print("Number of clusters: 3")
    print("Number of visitors:", visitor_features.count())
    print("Silhouette score:", silhouette)

    print("\nMACHINE LEARNING COMPLETED SUCCESSFULLY!")

finally:

    spark.stop()

    print("\nSpark session stopped.")
