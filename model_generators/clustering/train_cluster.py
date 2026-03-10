import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CORE_SAMPLE_PERCENTILE = 0.25

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

X = df[SEGMENT_FEATURES].to_numpy()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df["cluster_id"] = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_
# Sort by mean estimated_income (Column 0)
sorted_clusters = centers[:, 0].argsort()

cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium",
}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")

silhouette_avg = round(silhouette_score(X, df["cluster_id"]), 2)

cluster_distances = np.linalg.norm(X - centers[df["cluster_id"]], axis=1)
distance_cv = round((cluster_distances.std(ddof=1) / cluster_distances.mean()) * 100, 2)

core_distance_threshold = np.quantile(cluster_distances, CORE_SAMPLE_PERCENTILE)
core_mask = cluster_distances <= core_distance_threshold

refined_silhouette = round(
    silhouette_score(X[core_mask], df.loc[core_mask, "cluster_id"]), 2
)

refined_distance_cv = round(
    (cluster_distances[core_mask].std(ddof=1) / cluster_distances[core_mask].mean()) * 100,
    2,
)

cluster_summary = (
    df.groupby("client_class")[SEGMENT_FEATURES]
    .mean()
    .reset_index()
    .merge(
        df["client_class"]
        .value_counts()
        .rename_axis("client_class")
        .reset_index(name="count"),
        on="client_class",
    )
)

comparison_df = df[
    ["client_name", "estimated_income", "selling_price", "client_class"]
]


def predict_client_segment(estimated_income, predicted_price):
    cluster_id = kmeans.predict([[estimated_income, predicted_price]])[0]
    return cluster_mapping.get(cluster_id, "Unknown")


def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": distance_cv,
        "refined_silhouette": refined_silhouette,
        "refined_coefficient_variation": refined_distance_cv,
        "refined_sample_size": int(core_mask.sum()),
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

if __name__ == "__main__":
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Coefficient Variation: {distance_cv}%")
    print(f"Refined Silhouette: {refined_silhouette}")
    print(f"Refined CV: {refined_distance_cv}%")
    print(f"Refined Sample Size: {core_mask.sum()}")
