import firebase_admin
from firebase_admin import credentials, storage

class FirebaseStorageSingleton:
    __instance = None
    
    def __new__(cls):
        if FirebaseStorageSingleton.__instance is None:
            FirebaseStorageSingleton.__instance = object.__new__(cls)
            
            # Initialize Firebase app
            cred = credentials.Certificate("cerebralhacks-dba8c-firebase-adminsdk-tebp7-ae6980afe0.json")
            firebase_admin.initialize_app(cred)
            
            # Get a reference to the storage bucket
            FirebaseStorageSingleton.__instance.bucket = storage.bucket('cerebralhacks-dba8c.appspot.com')
        
        return FirebaseStorageSingleton.__instance

    def get_bucket(self):
        return self.bucket
