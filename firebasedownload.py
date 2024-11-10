import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase
cred = credentials.Certificate("cerebralhacks-dba8c-firebase-adminsdk-tebp7-ae6980afe0.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cerebralhacks-dba8c.appspot.com'
})

# Create a reference to the video file
bucket = storage.bucket()
blob = bucket.blob('scene.mp4')

# Download the video
blob.download_to_filename('scene.mp4')