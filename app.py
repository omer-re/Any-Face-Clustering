import os
import time
import shutil
import tempfile
import numpy as np
import streamlit as st
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages
import csv
import pickle  # For caching
import tqdm  # For terminal-based progress

# Define output directory and ensure it exists
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# File paths for caching and outputs
CACHE_FILE = os.path.join(OUTPUT_DIR, "face_clustering_cache.pkl")
CSV_FILENAME = os.path.join(OUTPUT_DIR, "face_clustering_results.csv")
CSV_HEADER = ["person", "original_filename"]

# Set page configuration for Streamlit
st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: grey;'>Any Face Clustering</h1>", unsafe_allow_html=True)

# File uploader with label
uploaded_files = st.file_uploader("Upload Images (png, jpg, jpeg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
no_of_files = len(uploaded_files)

# Initialize session state variables
if "data" not in st.session_state:
    st.session_state.data = []
if "data_arr" not in st.session_state:
    st.session_state.data_arr = None
if "encodings_arr" not in st.session_state:
    st.session_state.encodings_arr = None
if "labelIDs" not in st.session_state:
    st.session_state.labelIDs = None
if "numUniqueFaces" not in st.session_state:
    st.session_state.numUniqueFaces = 0
if "cluster_names" not in st.session_state:
    st.session_state.cluster_names = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "cluster" not in st.session_state:
    st.session_state.cluster = None

# Load cache if available and no new files are uploaded
if os.path.exists(CACHE_FILE) and no_of_files == 0 and not st.session_state.processed:
    with open(CACHE_FILE, "rb") as cache_file:
        cache_data = pickle.load(cache_file)
        st.session_state.data = cache_data["data"]
        st.session_state.data_arr = cache_data["data_arr"]
        st.session_state.encodings_arr = cache_data["encodings_arr"]
        st.session_state.labelIDs = cache_data["labelIDs"]
        st.session_state.numUniqueFaces = cache_data["numUniqueFaces"]
        st.session_state.cluster_names = cache_data.get("cluster_names", {})
    st.session_state.processed = True
    st.success("Loaded cached data successfully.")

# Perform face clustering only if new images are uploaded and data is not already present
if no_of_files > 0 and not st.session_state.processed:
    placeholder = st.empty()
    placeholder.success(f"{no_of_files} Images uploaded successfully!")
    time.sleep(3)
    placeholder.empty()

    # Create a single progress bar instance in Streamlit with a label
    progress_bar = st.progress(0, text="Processing uploaded images")

    # Process each uploaded file and use tqdm for console-based progress
    for idx, uploaded_file in enumerate(tqdm.tqdm(uploaded_files, desc="Processing uploaded images")):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image = cv2.imread(tfile.name)

        # Check if the image is loaded successfully
        if image is None:
            st.warning(f"Warning: Failed to load image {uploaded_file.name}. Skipping.")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the HOG model for faster processing without GPU
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Warn if no faces are found in this image
        if len(encodings) == 0:
            st.warning(f"Warning: No faces found in {uploaded_file.name}. Skipping.")
            continue

        # Add data for each detected face along with the original file name
        d = [{"imagePath": tfile.name, "originalFilename": uploaded_file.name, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        st.session_state.data.extend(d)

        # Update the single Streamlit progress bar with every iteration
        progress_bar.progress((idx + 1) / no_of_files)

    # Converting the data into a numpy array
    if len(st.session_state.data) > 0:
        st.session_state.data_arr = np.array(st.session_state.data)
    else:
        st.error("No valid face data was found in the uploaded images.")
        st.stop()

    # Extracting the 128-d facial encodings and placing them in a list
    if len(st.session_state.data_arr) > 0:
        st.session_state.encodings_arr = [item["encoding"] for item in st.session_state.data_arr]
    else:
        st.error("No valid encodings were found in the processed images.")
        st.stop()

    # Initialize clustering model only if there are valid encodings
    if st.session_state.encodings_arr and len(st.session_state.encodings_arr) > 0:
        cluster = DBSCAN(min_samples=3)
        cluster.fit(st.session_state.encodings_arr)

        # Ensure the `labels_` attribute exists
        if hasattr(cluster, 'labels_'):
            st.session_state.cluster = cluster
            st.session_state.labelIDs = np.unique(cluster.labels_)
            st.session_state.numUniqueFaces = len(np.where(st.session_state.labelIDs > -1)[0])
            st.session_state.processed = True
            st.balloons()

            # Save results to cache
            with open(CACHE_FILE, "wb") as cache_file:
                cache_data = {
                    "data": st.session_state.data,
                    "data_arr": st.session_state.data_arr,
                    "encodings_arr": st.session_state.encodings_arr,
                    "labelIDs": st.session_state.labelIDs,
                    "numUniqueFaces": st.session_state.numUniqueFaces,
                    "cluster_names": st.session_state.cluster_names,
                }
                pickle.dump(cache_data, cache_file)

# Display clustered results and allow naming clusters
if st.session_state.processed and st.session_state.cluster is not None:
    st.subheader(f"Number of unique faces identified (excluding the unknown faces): {st.session_state.numUniqueFaces}")

    # CSV data to be saved
    csv_data = []

    # Iterate through each cluster to manage labeling and exporting
    for labelID in st.session_state.labelIDs:
        # Check if cluster exists and labels are properly initialized before proceeding
        if st.session_state.cluster is not None and hasattr(st.session_state.cluster, 'labels_'):
            idxs = np.where(st.session_state.cluster.labels_ == labelID)[0]
        else:
            st.warning("Clustering data is missing. Please upload new images to process.")
            continue

        # Ensure valid indices exist for this label
        if len(idxs) == 0:
            st.warning(f"No images found for label {labelID}. Skipping.")
            continue

        # Set default cluster name and allow the user to rename it, update dynamically
        default_name = f"Face #{labelID + 1}" if labelID != -1 else "Unknown"
        cluster_name = st.text_input(f"Enter name for Cluster {labelID + 1}:", value=st.session_state.cluster_names.get(labelID, default_name), key=f"name_{labelID}")

        # Update the cluster name in session state whenever the input is changed
        if cluster_name != st.session_state.cluster_names.get(labelID):
            st.session_state.cluster_names[labelID] = cluster_name

        # Initialize the list of faces to include in the montage
        faces = []

        for idx in idxs:
            current_image = cv2.imread(st.session_state.data_arr[idx]["imagePath"])
            rgb_current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            (top, right, bottom, left) = st.session_state.data_arr[idx]["loc"]
            current_face = rgb_current_image[top:bottom, left:right]
            current_face = cv2.resize(current_face, (96, 96))
            faces.append(current_face)

            # Add entry to CSV data (use the original filename)
            csv_data.append([cluster_name, st.session_state.data_arr[idx]["originalFilename"]])

        # Create montage of face thumbnails, ensuring multiple images are displayed
        if faces:
            montages = build_montages(faces, (96, 96), (5, 5))
            st.write(cluster_name)
            for montage in montages:
                st.image(montage, caption=f"Cluster: {cluster_name}")

        # Commented out ZIP creation during development
        # if labelID != -1:
        #     dir_name = os.path.join(OUTPUT_DIR, cluster_name)
        #     if os.path.exists(dir_name):
        #         shutil.rmtree(dir_name)
        #     os.mkdir(dir_name)
        #     for i in idxs:
        #         face_image_name = f'image_{i}.jpg'
        #         face_image_path = os.path.join(dir_name, face_image_name)
        #         cv2.imwrite(face_image_path, cv2.imread(st.session_state.data_arr[i]["imagePath"]))
        #     zip_filename = os.path.join(OUTPUT_DIR, f"{cluster_name}.zip")
        #     shutil.make_archive(os.path.join(OUTPUT_DIR, cluster_name), 'zip', dir_name)
        #     # Delete the directory after creating the ZIP
        #     shutil.rmtree(dir_name)

    # Write CSV data to file every time the name is updated, using UTF-8 encoding
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_data)

    # Provide an option to download the CSV file
    with open(CSV_FILENAME, "rb") as fp:
        st.download_button(
            label="Download CSV of Face Clustering Results",
            data=fp,
            file_name=os.path.basename(CSV_FILENAME),
            mime="text/csv"
        )
