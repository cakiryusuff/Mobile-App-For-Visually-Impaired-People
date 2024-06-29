from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from numpy import array, uint8, frombuffer, zeros
from threading import Lock
from io import BytesIO
from ultralytics import YOLO
from PIL.Image import open
from utils import process_image
import face_recognition
import cv2
from firebase_database import write_data, write_log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the latest frame
latest_frame = None
frame_lock = Lock()
email_for_detect = None
trigger_for_database = True

def load_image_into_numpy_array(data):
    return array(open(BytesIO(data)))

@app.post("/register_face")
async def register_face(file: UploadFile, name: str = Form(...), email: str = Form(...)):
    # Yüz resmini al ve yüz encodlarını çıkar
    global email_for_detect
    global trigger_for_database
    image_data = await file.read()
    img = cv2.imdecode(frombuffer(image_data, uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings_list = face_recognition.face_encodings(rgb_img, face_locations)

    if len(face_encodings_list) == 0:
        raise HTTPException(status_code=400, detail="Yüz tespit edilemedi")

    face_encodings = face_encodings_list[0]
    face_names = name
    email_for_detect = email
    print(name)
    print(email)
    
    data = {
        "name": face_names,
        "face_encoding": str(face_encodings),
        "eposta": email
        }
    
    write_data(data)
    trigger_for_database = True
    
    return {"message": "Yüz başarıyla kaydedildi"}

@app.post("/upload_and_detect")
async def upload_and_detect(file: UploadFile = File(...), email: str = Form(...)):
    global latest_frame
    global trigger_for_database
    aa = await file.read()
    
    img = cv2.imdecode(frombuffer(aa, uint8), cv2.IMREAD_COLOR)
    
    detected_objects, answer_for_database = process_image(img, email, trigger_for_database)
    write_log(detected_objects, email=email)
    trigger_for_database = answer_for_database    
        
    with frame_lock:
        latest_frame = img
    
    return JSONResponse(content={"detected_objects": detected_objects})

@app.get("/video_feed")
def video_feed():
    def generate():
        global latest_frame
        while True:
                if latest_frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', latest_frame)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
