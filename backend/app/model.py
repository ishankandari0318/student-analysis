import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import os


# Load and prepare once at import time
df = pd.read_csv("data/student.csv")
df = df.dropna()
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

X = df[["studytime", "failures", "absences", "G1", "G2"]]
y = df["G3"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, df["pass"])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)


# Assign clusters to dataset
df["cluster"] = kmeans.predict(X_scaled)

# Calculate average G3 per cluster
cluster_means = df.groupby("cluster")["G3"].mean().sort_values()

# Map clusters to labels
cluster_labels = {}

labels = ["Low Performer", "Average Performer", "High Performer"]

for i, cluster_id in enumerate(cluster_means.index):
    cluster_labels[cluster_id] = labels[i]


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


#PREDICTION FUNCTION

def predict_student(input_data):
    try:
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order
        input_df = input_df[["studytime", "failures", "absences", "G1", "G2"]]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        return {
            "predicted_score": float(round(prediction, 2)),
            "pass": int(prediction >= 10)
        }

    except Exception as e:
        return {"error": str(e)}
    

#CLASSIFICATION FUNCTION

def classify_student(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[["studytime", "failures", "absences", "G1", "G2"]]

        input_scaled = scaler.transform(input_df)

        prediction = clf.predict(input_scaled)[0]

        return {
            "pass": int(prediction)
        }

    except Exception as e:
        return {"error": str(e)}
    
#CLUSTERING FUNCTION

def cluster_student(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[["studytime", "failures", "absences", "G1", "G2"]]

        input_scaled = scaler.transform(input_df)

        cluster = int(kmeans.predict(input_scaled)[0])

        return {
            "cluster": cluster,
            "label": cluster_labels[cluster]
        }

    except Exception as e:
        return {"error": str(e)}
    

#PCA


def generate_cluster_plot(input_data=None):
    try:
        clusters = kmeans.predict(X_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.5)

        # 🔥 Add user point if provided
        if input_data:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[["studytime", "failures", "absences", "G1", "G2"]]

            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)

            plt.scatter(
                input_pca[:, 0],
                input_pca[:, 1],
                color="red",
                s=120,
                label="Your Input"
            )

            plt.legend()

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("Student Clusters (PCA)")

        path = "saved_models/cluster_plot.png"
        plt.savefig(path)
        plt.close()

        return {"path": path}

    except Exception as e:
        return {"error": str(e)}