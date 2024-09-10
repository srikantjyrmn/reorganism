import firebase_admin
from firebase_admin import credentials, storage, firestore
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
# Set the path to your service account key file
# Initialize Firebase Admin SDK
def get_firebase_app(platform = 'firebase'):
    """Get or initialize Firebase app."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        # If running locally
        if platform == 'firebase':
            cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
            return firebase_admin.initialize_app(cred)
        else:
            cred = credentials.Certificate('./../reorganism-in-firebase-adminsdk-r7mr3-66c6f04d76.json')
            return firebase_admin.initialize_app()

app = get_firebase_app()
db = firestore.client()
bucket = storage.bucket('reorganism-in')

def read_vector_from_firestore(collection_name, document_id):
    """
    Read a vector from Firestore.
    
    :param collection_name: Name of the Firestore collection
    :param document_id: ID of the document
    :return: The stored vector (as numpy array) and metadata
    """
    doc_ref = db.collection(collection_name).document(document_id)
    doc = doc_ref.get()
    
    if doc.exists:
        data = doc.to_dict()
        vector = np.array(data['vector'])
        metadata = {k: v for k, v in data.items() if k != 'vector'}
        return vector, metadata
    else:
        return None, None

def vector_search(query, k=10):
    model = SentenceTransformer("hkunlp/instructor-large")
    vec = model.encode([[query, "represent the text for retrieval"]])
    collection = db.collection("my_docs")
    # Requires a single-field vector index
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(vec[0]),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    ).get()
    return [x.to_dict() for x in vector_query]

def read_from_firebase_storage(file_path, local_file_path):
    """
    Download a file from Firebase Storage.
    
    :param file_path: Path of the file in Firebase Storage
    :param local_file_path: Local path to save the downloaded file
    :return: True if successful, False otherwise
    """
    blob = bucket.blob(file_path)
    
    if not blob.exists():
        print(f"File {file_path} does not exist.")
        return False
    
    blob.download_to_filename(local_file_path)
    print(f"File downloaded to {local_file_path}")
    return True

def download_directory(directory_path, local_destination):
    """
    Download all files from a specified directory in Firebase Storage to a local destination.
    
    :param directory_path: Path of the directory in Firebase Storage
    :param local_destination: Local path to download the files to
    """
    files = list_files_in_directory(directory_path)
    
    for file_path in files:
        # Create the local directory structure if it doesn't exist
        local_file_path = os.path.join(local_destination, os.path.relpath(file_path, directory_path))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        read_from_firebase_storage(file_path, local_file_path)
