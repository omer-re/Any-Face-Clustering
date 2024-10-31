import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from ast import literal_eval  # Import literal_eval for safer evaluation
import time  # Import time to measure processing time

# Directory to save facial embeddings
EMBEDDINGS_FILE = "face_embeddings.json"
CLUSTERS_FILE = "face_clusters.csv"


# Function to extract facial embeddings from an image
def extract_embeddings(image_path):
    try:
        # Properly handle non-ASCII file paths and ensure correct path format
        norm_path = os.path.normpath(image_path)
        img_data = np.fromfile(norm_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return []

    if img is None:
        print(f"Warning: Unable to load image at {image_path}. Skipping.")
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use the HOG model for faster processing without GPU
    boxes = face_recognition.face_locations(img_rgb, model="hog")
    encodings = face_recognition.face_encodings(img_rgb, boxes)
    return encodings


# Function to load embeddings from existing file
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return pd.read_json(EMBEDDINGS_FILE)  # Load from JSON
    return pd.DataFrame(columns=["image", "embedding"])


# Function to save embeddings to JSON file
def save_embeddings(df):
    df.to_json(EMBEDDINGS_FILE, orient='records')  # Save as JSON


# Function to save clusters to CSV file
def save_clusters(df):
    df.to_csv(CLUSTERS_FILE, index=False)


# Function to cluster faces and create a dataframe with image associations
def cluster_faces(embeddings):
    embeddings_matrix = np.vstack(embeddings["embedding"].to_numpy())
    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_matrix_scaled = scaler.fit_transform(embeddings_matrix)

    # Adjusted DBSCAN parameters for better sensitivity
    clustering = DBSCAN(eps=0.5, min_samples=3, metric="cosine").fit(embeddings_matrix_scaled)
    embeddings["cluster"] = clustering.labels_
    return embeddings


# Function to process a folder of images and update the dataframe
def process_images(folder_path):
    start_time = time.time()  # Start time measurement

    embeddings = load_embeddings()
    new_data = []

    files = []
    for root, _, file_names in os.walk(folder_path):
        for file in file_names:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                files.append(os.path.join(root, file))

    for image_path in tqdm(files, desc="Processing images"):
        if not embeddings[embeddings['image'] == image_path].empty:
            continue
        image_embeddings = extract_embeddings(image_path)
        if image_embeddings:
            for emb in image_embeddings:
                new_data.append({"image": image_path, "embedding": emb.tolist()})

    new_embeddings_df = pd.DataFrame(new_data)
    if not new_embeddings_df.empty:
        embeddings = pd.concat([embeddings, new_embeddings_df], ignore_index=True)
        embeddings = cluster_faces(embeddings)
        save_embeddings(embeddings)
        save_clusters(embeddings[["image", "cluster"]])

    end_time = time.time()  # End time measurement
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


# Function to find the closest cluster for a new face
def find_closest_cluster(new_image_path):
    embeddings = load_embeddings()
    if embeddings.empty:
        print("No existing embeddings to compare.")
        return None

    new_embeddings = extract_embeddings(new_image_path)
    if not new_embeddings:
        print("No face detected in the image.")
        return None

    new_embedding = new_embeddings[0]  # Corrected to properly access the embedding
    embeddings_matrix = np.vstack(embeddings["embedding"].to_numpy())

    min_dist = float('inf')
    closest_cluster = -1
    for cluster in tqdm(embeddings['cluster'].unique(), desc="Finding closest cluster"):
        cluster_embeddings = embeddings[embeddings['cluster'] == cluster]
        cluster_centroid = np.mean(np.vstack(cluster_embeddings["embedding"].to_numpy()), axis=0)
        dist = distance.cosine(new_embedding, cluster_centroid)
        if dist < min_dist:
            min_dist = dist
            closest_cluster = cluster
    return closest_cluster


# Function to generate a report with thumbnails of each cluster
def generate_cluster_report():
    embeddings = load_embeddings()
    if embeddings.empty:
        print("No embeddings available to generate report.")
        return

    clusters = embeddings['cluster'].unique()
    fig, axs = plt.subplots(len(clusters), 1, figsize=(15, len(clusters) * 5))
    if len(clusters) == 1:
        axs = [axs]

    for idx, cluster in enumerate(tqdm(clusters, desc="Generating cluster report")):
        if cluster == -1:
            continue  # Skip noise points
        cluster_data = embeddings[embeddings['cluster'] == cluster]
        num_images = len(cluster_data)
        n = int(np.ceil(np.sqrt(num_images)))  # Calculate grid size

        sub_fig, sub_axes = plt.subplots(n, n, figsize=(15, 15))
        sub_fig.suptitle(f'Cluster {cluster}', fontsize=16)

        sub_axes = sub_axes.flatten()  # Flatten to easily iterate
        for i, (_, row) in enumerate(cluster_data.iterrows()):
            image_path = row['image']
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Unable to load image at {image_path}. Skipping.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sub_axes[i].imshow(img_rgb)
            sub_axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(sub_axes)):
            sub_axes[j].axis('off')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        axs[idx].imshow(np.zeros((15, 15, 3), dtype=np.uint8))  # Placeholder for tab navigation
        axs[idx].axis('off')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Ensure the folder path is properly formatted
    folder_to_process = r"G:\My Drive\OMER_PERSONAL\Omer\Gallery\Wedding\magnets\test"
    process_images(folder_to_process)
    print("Clusters updated.")

    # Uncomment to find the closest cluster
    # new_image = r"path/to/new_image.jpg"
    # cluster = find_closest_cluster(new_image)
    # print(f"Closest cluster for the new image is: {cluster}")

    # Generate a report with thumbnails of each cluster
    generate_cluster_report()
