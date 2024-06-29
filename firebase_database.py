import firebase_admin
from firebase_admin import db, credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import numpy as np
import pytz
from pytz import timezone
from datetime import datetime

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

data = {
    "name":"wash",
    "face_encoding":"0.123123123, 0.12342354",
    "eposta":"cakir_yusuff@outlook.com"
    }

def write_data(dataa):
    doc_ref = db.collection("faceEncodings").document()
    doc_ref.set(dataa)
    
def write_log(data, email):
    user_collection = db.collection("users").document(email).collection("detections")
    
    utc_now = datetime.now()
    utc = pytz.timezone('UTC')
    aware_date = utc.localize(utc_now)
    turkey = timezone('Europe/Istanbul')
    now_turkey = aware_date.astimezone(turkey).strftime('%Y-%m-%d %H:%M:%S')
    
    user_collection.document(str(utc_now)).set({
            "object": data,
            "timestamp": str(utc_now),
        })

    
def get_documents_with_status(collection_name, status_value):
    try:
        data_list = []
        encoded_faces = []
        
        doc_ref = db.collection(collection_name)
        #make your query
        query = doc_ref.where(filter=FieldFilter("eposta", "==", status_value))
        #stream for results
        docs = query.stream()
        for doc in docs:
            data = doc.to_dict()
            data_list.append(data)
            #print("Document data:", data)
            #print("\n")

        for list in data_list:
            data_list = list["face_encoding"].strip('[]').split()
            int_list = [[float(item) for item in data_list]]
            int_list = np.array(int_list)
            encoded_faces.append((int_list, list["name"]))
                        
        return encoded_faces
    
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")


#READ DATA
#print(get_all_docs("faceEncodings"))
#print(get_document('tasksCollection','flfWqunWhohtaJ7OpSzO' ))
#lists = get_documents_with_status("faceEncodings", "cakir_yusauff@outlook.com")
# get_different_status("tasksCollection", "TODO", "done")



    

"""
data_list2 = aaa[1]["face_encoding"].strip('[]').split()

# Convert each string to a float, then to an int
int_list2 = [[float(item) for item in data_list]]

int_list2 = np.array(int_list2)

matches = face_recognition.compare_faces(int_list, int_list2)
print(matches)
"""