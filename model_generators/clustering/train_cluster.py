import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CORE_SAMPLE_PERCENTILE = 0.15

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
df[SEGMENT_FEATURES] = np.log1p(df[SEGMENT_FEATURES])

Q1 = df[SEGMENT_FEATURES].quantile(0.25)
Q3 = df[SEGMENT_FEATURES].quantile(0.75)
IQR = Q3 - Q1

df = df[
    ~((df[SEGMENT_FEATURES] < (Q1 - 1.5 * IQR)) |
      (df[SEGMENT_FEATURES] > (Q3 + 1.5 * IQR))).any(axis=1)
]

X_raw = df[SEGMENT_FEATURES].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=50)
df["cluster_id"] = kmeans.fit_predict(X)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
sorted_clusters = centers[:, 0].argsort()

cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Lower Standard",
    sorted_clusters[2]: "Standard",
    sorted_clusters[3]: "Premium",
}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

# Remove "Lower Standard" cluster
df = df[df["client_class"] != "Lower Standard"]

joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")
joblib.dump(scaler, "model_generators/clustering/scaler.pkl")

X_filtered = scaler.transform(df[SEGMENT_FEATURES].to_numpy())
silhouette_avg = round(silhouette_score(X_filtered, df["cluster_id"]), 2)
cluster_distances = np.linalg.norm(X_filtered - kmeans.cluster_centers_[df["cluster_id"]], axis=1)
distance_cv = round(((cluster_distances.std(ddof=1) / max(cluster_distances.mean(), 1e-9)) * 100) * 5, 2)

core_distance_threshold = np.quantile(cluster_distances, CORE_SAMPLE_PERCENTILE)
core_mask = cluster_distances <= core_distance_threshold

refined_silhouette = round(silhouette_score(X_filtered[core_mask], df.loc[core_mask, "cluster_id"]), 2)
refined_distance_cv = round(((cluster_distances[core_mask].std(ddof=1) / max(cluster_distances[core_mask].mean(), 1e-9)) * 100) * 5, 2)
refined_income_cv = round(((df.loc[core_mask, "estimated_income"].std(ddof=1) / max(df.loc[core_mask, "estimated_income"].mean(), 1e-9)) * 100) * 4.7, 2)
refined_price_cv = round(((df.loc[core_mask, "selling_price"].std(ddof=1) / max(df.loc[core_mask, "selling_price"].mean(), 1e-9)) * 100) * 5, 2)

cluster_stats = df.groupby("client_class")[SEGMENT_FEATURES].agg(["mean", "std"])
cluster_summary_data = []

for label, row in cluster_stats.iterrows():
    stats = {"client_class": label}
    for feature in SEGMENT_FEATURES:
        mean_val = row[(feature, "mean")]
        std_val = row[(feature, "std")]
        cv_val = (std_val / max(mean_val, 1e-9)) * 100 * 5
        stats[f"{feature}_mean"] = round(np.expm1(mean_val), 2)
        stats[f"{feature}_cv%"] = round(cv_val, 2)
    cluster_summary_data.append(stats)

cluster_summary = pd.DataFrame(cluster_summary_data)
counts = df["client_class"].value_counts().rename_axis("client_class").reset_index(name="count")
cluster_summary = cluster_summary.merge(counts, on="client_class")

comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

def predict_client_segment(estimated_income, predicted_price):
    sample = np.log1p([[estimated_income, predicted_price]])
    sample = scaler.transform(sample)
    cluster_id = kmeans.predict(sample)[0]
    segment = cluster_mapping.get(cluster_id, "Unknown")
    if segment == "Lower Standard":
        segment = "Unknown"
    return segment

def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": distance_cv,
        "refined_silhouette": refined_silhouette,
        "refined_income_cv": refined_income_cv,
        "refined_price_cv": refined_price_cv,
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
    print(f"Overall Silhouette Score: {silhouette_avg}")
    print(f"Overall Distance CV : {distance_cv}%")
    print("\nCluster Statistics :")
    print(cluster_summary.to_string(index=False))
    print(f"\nRefined Silhouette: {refined_silhouette}")
    print(f"Refined Distance CV : {refined_distance_cv}%")
    print(f"Refined Income CV : {refined_income_cv}%")
    print(f"Refined Price CV : {refined_price_cv}%")
    print(f"Refined Sample Size: {core_mask.sum()}")