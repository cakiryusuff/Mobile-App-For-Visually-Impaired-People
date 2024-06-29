from numpy import array, uint8, zeros
import cv2
from torch.hub import load
from torch import no_grad
from torch.nn.functional import interpolate
from ultralytics import YOLO
import yaml
import face_recognition
import firebase_database

with open("config.yml", "r") as file:
    yaml_file = yaml.safe_load(file)

#model_type = yaml_file["midas"]["model_weight"]
#midas = load("intel-isl/MiDaS", model_type)
#midas.eval()

#midas_transforms = load("intel-isl/MiDaS", "transforms")
#transform = midas_transforms.dpt_transform

yolo_model = YOLO(yaml_file["yolo"]["model_weight"])

turkce_dict = {
    'person': 'İnsan',
    'bicycle': 'Bisiklet',
    'car': 'Araba',
    'motorcycle': 'Motosiklet',
    'airplane': 'Uçak',
    'bus': 'Otobüs',
    'train': 'Tren',
    'truck': 'Kamyon',
    'boat': 'Tekne',
    'traffic light': 'Trafik ışığı',
    'fire hydrant': 'Yangın musluğu',
    'stop sign': 'Dur işareti',
    'parking meter': 'Parkomat',
    'bench': 'Bank',
    'bird': 'Kuş',
    'cat': 'Kedi',
    'dog': 'Köpek',
    'horse': 'At',
    'sheep': 'Koyun',
    'cow': 'İnek',
    'elephant': 'Fil',
    'bear': 'Ayı',
    'zebra': 'Zebra',
    'giraffe': 'Zürafa',
    'backpack': 'Sırt çantası',
    'umbrella': 'Şemsiye',
    'handbag': 'El çantası',
    'tie': 'Kravat',
    'suitcase': 'Bavul',
    'frisbee': 'Frizbi',
    'skis': 'Kayaklar',
    'snowboard': 'Snowboard',
    'sports ball': 'Spor topu',
    'kite': 'Uçurtma',
    'baseball bat': 'Beyzbol sopası',
    'baseball glove': 'Beyzbol eldiveni',
    'skateboard': 'Kaykay',
    'surfboard': 'Sörf tahtası',
    'tennis racket': 'Tenis raketi',
    'bottle': 'Şişe',
    'wine glass': 'Şarap kadehi',
    'cup': 'Kupa',
    'fork': 'Çatal',
    'knife': 'Bıçak',
    'spoon': 'Kaşık',
    'bowl': 'Kase',
    'banana': 'Muz',
    'apple': 'Elma',
    'sandwich': 'Sandviç',
    'orange': 'Portakal',
    'broccoli': 'Brokoli',
    'carrot': 'Havuç',
    'hot dog': 'Sosisli sandviç',
    'pizza': 'Pizza',
    'donut': 'Donut',
    'cake': 'Pasta',
    'chair': 'Sandalye',
    'couch': 'Kanepe',
    'potted plant': 'Saksı bitkisi',
    'bed': 'Yatak',
    'dining table': 'Yemek masası',
    'toilet': 'Tuvalet',
    'tv': 'Televizyon',
    'laptop': 'Dizüstü bilgisayar',
    'mouse': 'Fare',
    'remote': 'Uzaktan kumanda',
    'keyboard': 'Klavye',
    'cell phone': 'Cep telefonu',
    'microwave': 'Mikrodalga fırın',
    'oven': 'Fırın',
    'toaster': 'Ekmek kızartma makinesi',
    'sink': 'Lavabo',
    'refrigerator': 'Buzdolabı',
    'book': 'Kitap',
    'clock': 'Saat',
    'vase': 'Vazo',
    'scissors': 'Makas',
    'teddy bear': 'Oyuncak ayı',
    'hair drier': 'Saç kurutma makinesi',
    'toothbrush': 'Diş fırçası'
}


def crop_image(image):
    height, width = image.shape[:2]

    # Kırmızı çizgilerin koordinatları
    points = array([
        [int(width * 0.1), height],  # Sol alt köşe
        [int(width * 0.2), 0],       # Sol üst köşe
        [int(width * 0.8), 0],       # Sağ üst köşe
        [int(width * 0.9), height]   # Sağ alt köşe
    ])
    mask = zeros((height, width), dtype=uint8)
    cv2.fillPoly(mask, [points], 255)
    image[mask == 0] = [0, 0, 0]
    
    return image

"""
def yolo_predict(img):
    image = crop_image(img)
    results = model.predict(image, verbose = False)
    detected_objects = []
    for result in results[0]:
        class_id = result.boxes.cpu().numpy()
        value = int(class_id.cls)
        detected_objects.append(model.names[value])
        
    return detected_objects
"""
"""
def get_depth_map(image_rgb):
    input_batch = transform(image_rgb)
    with no_grad():
        prediction = midas(input_batch)
    prediction = interpolate(
        prediction.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map
"""

def process_image(image, email_for_detect, trigger_for_database, near_threshold=500, far_threshold=150):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_image = crop_image(img_rgb)
    results = yolo_model.predict(cropped_image, verbose = False)

    #depth_map = get_depth_map(img_rgb)

    distance = []
    for result in results:
        for bbox in result.boxes:
            class_id = int(bbox.cls)
            x1, y1, x2, y2 = map(int, bbox.xyxy[0].cpu().numpy())
                
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{yolo_model.names[class_id]}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ################ FOR FACE RECOGNATION ##################
            if yolo_model.names[class_id] == "person":
                name = "Bilinmeyen kişi"
                if trigger_for_database:
                    encoded_faces = firebase_database.get_documents_with_status("faceEncodings", email_for_detect)
                    if len(encoded_faces) != 0:
                        trigger_for_database = True
                        face_locations = face_recognition.face_locations(img_rgb)
                        if len(face_locations) != 0:
                            face_encodings_list = face_recognition.face_encodings(img_rgb, face_locations)
                            
                            for face_encoding in encoded_faces:
                                matches = face_recognition.compare_faces(face_encoding[0], face_encodings_list)

                                if True in matches:
                                    distance.append(f'{face_encoding[1]}')
                                else:
                                    distance.append(name)
                    else:
                        trigger_for_database = False
                        distance.append(name)
                else:  
                    distance.append(name)

            objects = f'{turkce_dict[yolo_model.names[class_id]]}'
            if yolo_model.names[class_id] == "person":
                pass
            else:
                distance.append(objects)
            
            #######################################################
            
    #if len(distance) == 0:
    #    distance.append("None")
            

    return distance, trigger_for_database