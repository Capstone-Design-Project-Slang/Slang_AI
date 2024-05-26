from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

with open("lgbm_model_sports.pkl", "rb") as f:
    clf_sports = pickle.load(f)

with open("lgbm_model_korean.pkl", "rb") as f:
    clf_korean = pickle.load(f)
with open("lgbm_model_animals.pkl", "rb") as f:
    clf_animals = pickle.load(f)
with open("lgbm_model_food.pkl", "rb") as f:
    clf_food = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

def model(hand, clf, flag):

    if flag == 0:
        actions = ['ㄱ', 'ㄴ','ㄷ', 'ㄹ', 'ㅁ','ㅂ', 'ㅅ', 'ㅇ', 'ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㅏ','ㅑ', 'ㅓ','ㅕ', 'ㅗ','ㅛ','ㅜ','ㅠ', 'ㅡ', 'ㅣ', 'ㅐ','ㅔ','ㅚ','ㅟ','ㅒ','ㅖ','ㅢ' ]
    elif flag == 1:
        actions = ['축구','야구', '농구', '배구', '탁구', '테니스', '태권도', '씨름', '유도', '수영', '스케이트', '스키', '사격', '팬싱', '검도' ]
    elif flag == 2:
        actions = ['기린','하마', '낙타', '사슴', '곰', '코뿔소', '늑대', '고양이', '거북이', '악어', '고릴라', '쥐', '소', '범', '토끼', '양', '닭', '개', '돼지']
    elif flag == 3:
        actions = ['먹다','쌀', '고구마', '감자', '고기', '달걀', '두부', '배', '포도', '수박', '과자', '빵', '차', '술', '요리', '맛있다', '배고프다', '배부르다']

    for res in hand:
        joint = np.zeros((21, 4))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
        
        v1 = joint[[0,1,2,3,0,5,6,7,5,0,9,10,11,9,0,13,14,15,13,0,17,18,19], :3] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,9,10,11,12,13,13,14,15,16,17,17,18,19,20], :3] # Child joint
        v = v2 - v1 # [23, 3]

        # 20개의 관절 정보를 정규화한다
        # (손가락 관절의 길이와 상관 없이 인식할 수 있도록)
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,4,8,9,10,11,9,13,14,15,16,14,18,19,20,21],:], 
        v[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],:])) # [21,]

        angle = np.degrees(angle) # Convert radian to degree
        d = np.concatenate([joint[:,3], v.flatten(), angle])
        input_data = np.expand_dims(d, axis=0)

        # y_pred = 예측값
        y_pred = actions[int(clf.predict(input_data)[0])]
    return y_pred
    

def gen(camera):
    while True:
        _, img = camera.read()
        if img is None:
            continue
        img = cv2.flip(img, 1)
        result = hands.process(img)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    

@app.route("/video_sports")
def video_sports():
    return Response(gen(cap), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_korean")
def video_korean():
    return Response(gen(cap), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_animals")
def video_animals():
    return Response(gen(cap), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_food")
def video_food():
    return Response(gen(cap), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/prediction_sports", methods=["GET"])
def prediction_sports():
    ret, img = cap.read()
    if not ret:
        return "Error!"
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        return str(model(result.multi_hand_landmarks, clf_sports, 1))
    return "Error!"

@app.route("/prediction_korean", methods=["GET"])
def prediction_korean():
    ret, img = cap.read()
    if not ret:
        return "Error!"
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        return str(model(result.multi_hand_landmarks, clf_korean, 0))
    return "Error!"

@app.route("/prediction_animals", methods=["GET"])
def prediction_animals():
    ret, img = cap.read()
    if not ret:
        return "Error!"
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        return str(model(result.multi_hand_landmarks, clf_animals, 2))
    return "Error!"

@app.route("/prediction_food", methods=["GET"])
def prediction_food():
    ret, img = cap.read()
    if not ret:
        return "Error!"
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        return str(model(result.multi_hand_landmarks, clf_food, 3))
    return "Error!"


if __name__ == "__main__":
    app.run(debug=True)
