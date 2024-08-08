import csv
import queue
from flask import Flask, render_template, request, Response, jsonify, flash, url_for, redirect
from flask_login import UserMixin, LoginManager, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import threading
import librosa
import json
import time
import math
import torch
import os
import io
from .MER_model import RCNN, DynamicPCALayer_Seq
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

def Processor_Creation():
    app = Flask(__name__)
    app.config['Password'] = 'UserPassword'
    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.init_app(app)

    # Initialize Firestore
    cred = credentials.Certificate('phonic-botany-428915-s3-47cc5c28ee4c.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    app.config['SECRET_KEY'] = '19980706'

    # GCS
    bucket_name = 'music-visualisation-app'
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} exists.")
    except Exception as e:
        print(f"Bucket {bucket_name} does not exist. Creating new bucket.")
        bucket = storage_client.create_bucket(bucket_name)
        print(f"Bucket {bucket_name} created.")

    # Ensure users collection exists
    def ensure_users_collection():
        try:
            # Attempt to get a document from the users collection
            user_ref = db.collection('users').document('dummy')
            user_ref.set({'check': True})  # This will create the collection if it doesn't exist
            user_ref.delete()  # Clean up the dummy document
            print("Users collection is ready.")
        except Exception as e:
            print(f"Error ensuring users collection: {e}")

    ensure_users_collection()

    from .user_class import User

    @login_manager.user_loader
    def load_user(user_id):
        user_ref = db.collection('users').document(user_id)
        user = user_ref.get()
        if user.exists:
            user_data = user.to_dict()
            return User(
                user_id=user.id,
                user_email=user_data['user_email'],
                user_name=user_data['user_name'],
                user_password=user_data['user_password'],
                user_age=user_data.get('user_age'),
                user_gender=user_data.get('user_gender'),
                user_hue_base=user_data.get('user_hue_base'),
                user_sat_base=user_data.get('user_sat_base'),
                user_lig_base=user_data.get('user_lig_base')
            )
        return None

    long_term_store = []
    clients = []
    T_sending = []
    T_receiving = []
    T_display = []
    emotion_source = []
    audio_memory = []
    feedback_value = {'value':""}
    X_recording = []
    Y_recording = []
    Y_c = {"value" : [0, 0]}
    processing_event = threading.Event()
    mer_event = threading.Event()
    simulator = threading.Event()
    stop_event = threading.Event()
    lock = threading.Lock()
    model = RCNN()
    model_path = 'Back_Stage/best_model_10s_100.pth'
    pca_paths = ['Back_Stage/pca1_10s_100.pkl']
    model.load_model(model_path, pca_paths)
    model.eval()

    @app.route('/')
    def index():
        return render_template('start.html')

    @app.route('/main')
    @login_required
    def main():
        return render_template('C_index.html', user=current_user)

    @app.route('/login')
    def login():
        return render_template('login.html')

    @app.route('/login', methods=['POST'])
    def login_post():
        USER_email = request.form['user_email']
        USER_password = request.form['user_password']
        user_ref = db.collection('users').where('user_email', '==', USER_email).limit(1).stream()
        USER = None
        for doc in user_ref:
            USER = doc.to_dict()
            USER['id'] = doc.id
            break

        if not USER or not check_password_hash(USER['user_password'], USER_password):
            flash('Please check your e-mail address or password.')
            return redirect(url_for('login'))

        user_obj = User(
            user_id=USER['id'],
            user_email=USER['user_email'],
            user_name=USER['user_name'],
            user_password=USER['user_password'],
            user_age=USER.get('user_age'),
            user_gender=USER.get('user_gender'),
            user_hue_base=USER.get('user_hue_base'),
            user_sat_base=USER.get('user_sat_base'),
            user_lig_base=USER.get('user_lig_base')
        )
        login_user(user_obj)
        return redirect(url_for('main'))

    @app.route('/signup')
    def signup():
        return render_template('signup.html')

    @app.route('/signup', methods=['POST'])
    def signup_post():
        USER_email = request.form.get('user_email')
        USER_password = request.form.get('user_password')
        USER_name = request.form.get('user_name')

        user_ref = db.collection('users').where('user_email', '==', USER_email).limit(1).stream()
        USER = None
        for doc in user_ref:
            USER = doc.to_dict()
            break

        if USER:
            flash('Email address already exists')
            return redirect(url_for('signup'))

        NEW_USER = {
            'user_email': USER_email,
            'user_name': USER_name,
            'user_password': generate_password_hash(USER_password, method='pbkdf2:sha256'),
            'user_age': 25,
            'user_gender': 'X',
            'user_hue_base': [240, 60, 0, 120, 180],
            'user_sat_base': [40, 45, 55, 60, 50],
            'user_lig_base': [40, 45, 55, 60, 50]
        }

        user_ref = db.collection('users').add(NEW_USER)
        user_id = user_ref[1].id
        NEW_USER['id'] = user_id

        user_obj = User(
            user_id=user_id,
            user_email=NEW_USER['user_email'],
            user_name=NEW_USER['user_name'],
            user_password=NEW_USER['user_password'],
            user_age=NEW_USER.get('user_age'),
            user_gender=NEW_USER.get('user_gender'),
            user_hue_base=NEW_USER.get('user_hue_base'),
            user_sat_base=NEW_USER.get('user_sat_base'),
            user_lig_base=NEW_USER.get('user_lig_base')
        )
        login_user(user_obj)
        return redirect(url_for('filling'))

    @app.route('/filling')
    @login_required
    def filling():
        return render_template('filling.html', user=current_user)

    @app.route('/filling', methods=['POST'])
    @login_required
    def filling_post():
        user_ref = db.collection('users').document(current_user.user_id)
        user_ref.update({
            'user_age': int(request.form.get('user_age')),
            'user_gender': request.form.get('user_gender'),
            'user_hue_base': [
                int(request.form.get('lvla_H')), int(request.form.get('hvla_H')), int(request.form.get('lvha_H')),
                int(request.form.get('hvha_H')), int(request.form.get('nn_H'))
            ],
            'user_sat_base': [
                int(request.form.get('lvla_S')), int(request.form.get('hvla_S')), int(request.form.get('lvha_S')),
                int(request.form.get('hvha_S')), int(request.form.get('nn_S'))
            ],
            'user_lig_base': [
                int(request.form.get('lvla_L')), int(request.form.get('hvla_L')), int(request.form.get('lvha_L')),
                int(request.form.get('hvha_L')), int(request.form.get('nn_L'))
            ]
        })
        return redirect(url_for('main'))

    @app.route('/logout')
    @login_required
    def logout():
        long_term_store.clear()
        stop_event.set()
        logout_user()
        return redirect(url_for('index'))

    def send_to_clients(data):
        dead_clients = []
        for client in clients:
            try:
                client.put(data)
            except Exception as e:  # 如果发送失败，假设客户端已断开
                dead_clients.append(client)
        for client in dead_clients:
            clients.remove(client)

    @app.route('/register_client')
    @login_required
    def register_client():
        def gen():
            q = queue.Queue()
            clients.append(q)

            try:
                while True:
                    try:
                        data = q.get(timeout=3000)  # 设置超时以避免长时间阻塞

                        yield data

                    except queue.Empty:
                        # 如果超时没有数据，发送一个保持连接的心跳信号
                        yield 'data: {}\n\n'  # 发送空数据包来保持连接
            except GeneratorExit:
                # 当客户端断开连接时，清理操作
                clients.remove(q)

        return Response(gen(), mimetype='text/event-stream')

    def database_upload(user_email):
        T = time.time()
        X = torch.stack(X_recording, dim=0)
        Y = torch.stack(Y_recording, dim=0)
        E = feedback_value['value']
        blob_path_X = f"{user_email}/{T}/X.pt"
        blob_path_Y = f"{user_email}/{T}/Y.pt"
        blob_path_E = f"{user_email}/{T}/E.txt"
        X_stream = io.BytesIO()
        Y_stream = io.BytesIO()
        E_stream = io.BytesIO()
        torch.save(X, X_stream)
        X_stream.seek(0)
        torch.save(Y, Y_stream)
        Y_stream.seek(0)
        E_stream.write(E.encode('utf-8'))
        E_stream.seek(0)
        blob_X = bucket.blob(blob_path_X)
        blob_Y = bucket.blob(blob_path_Y)
        blob_E = bucket.blob(blob_path_E)
        blob_X.upload_from_file(X_stream)
        blob_Y.upload_from_file(Y_stream)
        blob_E.upload_from_file(E_stream, content_type='text/plain')
        min_length = min(len(T_sending), len(T_receiving), len(T_display))
        csv_stream = io.BytesIO()
        text_stream = io.TextIOWrapper(csv_stream, encoding='utf-8', newline='')
        csv_writer = csv.writer(text_stream)

        for i in range(min_length):
            csv_writer.writerow([T_sending[i], T_receiving[i], T_display[i]])

        text_stream.flush()
        csv_stream.seek(0)
        blob_path_TD = f"{user_email}/{T}/Time_recording.csv"
        blob_TD = bucket.blob(blob_path_TD)
        blob_TD.upload_from_file(csv_stream, content_type='text/csv')

        print("uploaded")
        print(len(T_sending), len(T_receiving), len(T_display))



    def MER():
        print("MER Started")
        while True:
            mer=0
            Start_time=time.time()
            if stop_event.is_set():
                return {"status": "Stopped"}, 200

            with lock:
                l = len(emotion_source)
                print("MER", l)
                if l >= 441000:
                    mer=1
                    ess = np.array(emotion_source[-441000:])
                    del emotion_source[:-44100*9]
                elif l >= 44100:
                    mer=1
                    ess = np.zeros(441000)
                    ess[:l] = np.array(emotion_source)
            if mer==1:
                S = librosa.feature.melspectrogram(y=ess, sr=44100, n_mels=26, hop_length=441)
                S_dB = librosa.power_to_db(S, ref=np.max)
                zcr = librosa.feature.zero_crossing_rate(ess, hop_length=441)
                mfccs = librosa.feature.mfcc(y=ess, sr=44100, n_mfcc=26, hop_length=441)
                F = torch.tensor(np.stack([np.vstack((S_dB, zcr, mfccs))], axis=0))
                F = [F[:, :, i * 100:i * 100 + 100] for i in range(10)]
                F = torch.tensor(np.stack(F, axis=1))
                X_recording.append(F)
                Y = model(F.float())
                Y_recording.append(Y[0, :])
                with lock:
                    Y_c["value"] = Y[0, :].tolist()
                    print(Y_c)
            time.sleep(1-(Start_time-time.time()))

    def process_data(user_email, user_hue_base, user_sat_base, user_lig_base):
        print("Process started")
        T_start = time.time()
        count_T = 0
        pitch_record = np.zeros(128)
        pitch_mag = np.zeros(128)
        pitch_id = np.zeros(128)
        time_record = 0.0000001
        middle = [200, 200]
        ID = 1
        a = 0
        EC=0
        Ycc = Y_c['value']
        pitch_active = np.zeros(128)
        note_pic = []
        angle = np.linspace(0, 2 * math.pi, 360)
        radius = np.linspace(0, 200, 128)
        print("process cycle start")
        while True:
            if stop_event.is_set():
                if feedback_value['value'] != "cancel":
                    database_upload(user_email)
                print("sss", feedback_value['value'])
                T = time.time()
                print('Stop time:', T)
                print("Stop event is set. Performing cleanup and exiting.")
                stop_event.clear()
                processing_event.clear()
                mer_event.clear()
                return {"status": "Stopped"}, 200
            with lock:
                l = len(long_term_store)
            if l >= 4410:
                with lock:
                    short_term_store = long_term_store[:4410]
                    del long_term_store[:4410]
                    T_receiving.append(time.time())
                    emotion_source.extend(short_term_store)
                EC+=1
                #print("cut", l, len(long_term_store))
                pitches, magnitudes = librosa.piptrack(y=np.array(short_term_store), sr=44100, hop_length=441, threshold=0.5)
                pitch_times = librosa.times_like(pitches, sr=44100, hop_length=441)
                if EC==10:
                    with lock:
                        Ycc = Y_c['value']
                    print(Ycc)
                    EC = 0
                for j in range(pitches.shape[1]):
                    current_time = pitch_times[j] + time_record
                    for i in range(pitches.shape[0]):
                        if magnitudes[i, j] > 0:
                            midi_note = int(librosa.hz_to_midi(pitches[i, j]))
                            pitch_active[midi_note] = 1
                            if pitch_mag[midi_note] < magnitudes[i, j]:
                                pitch_mag[midi_note] = magnitudes[i, j]

                    if j % pitches.shape[1] == 0:  # 每0.1s更新音符圆的信息
                        count_T += 0.1
                        for p in range(128):
                            if pitch_active[p] == 0 and pitch_record[p] != 0:
                                pitch_id[p] = 0
                                pitch_record[p] = 0

                            if pitch_active[p] != 0 and pitch_record[p] != 0:
                                for element in note_pic:
                                    if element["pitch"] == p:
                                        element["life"] += 1

                        for element in note_pic:
                            element["size"] += 0.5
                            element["life"] -= 1
                            element["opacity"] = 1 - (10 - element["life"]) * 0.1
                        note_pic = [item for item in note_pic if item['life'] > 0]
                        for p in range(128):
                            if pitch_active[p] != 0 and pitch_record[p] == 0 and current_time - pitch_record[p] > 0.05:
                                if a < 360:
                                    angle_N = angle[a]
                                    a += 1
                                else:
                                    a -= 360
                                    angle_N = angle[a]
                                    a += 1
                                radius_N = radius[p]
                                x = middle[0] + radius_N * math.cos(angle_N)
                                y = middle[1] + radius_N * math.sin(angle_N)
                                if ((Ycc[0] < 0) & (Ycc[1] < 0)) & ((Ycc[0] < -0.1) | (Ycc[1] < -0.1)):
                                    s = 0  # Sad / Bored
                                elif ((Ycc[0] < 0) & (Ycc[1] > 0)) & ((Ycc[0] < -0.1) | (Ycc[1] > 0.1)):
                                    s = 1  # Content / Relaxed
                                elif ((Ycc[0] > 0) & (Ycc[1] < 0)) & ((Ycc[0] > 0.1) | (Ycc[1] < -0.1)):
                                    s = 2  # Angry / Frustrated
                                elif ((Ycc[0] > 0) & (Ycc[1] > 0)) & ((Ycc[0] > 0.1) | (Ycc[1] > 0.1)):
                                    s = 3  # Excited / Happy
                                else:
                                    s = 4  # neutral

                                Emotion = "Sad / Bored" if s == 0 else "Content / Relaxed" if s == 1 else "Angry / Frustrated" if s == 2 else "Excited / Happy" if s == 3 else "neutral"

                                Hue_Base = user_hue_base
                                Saturation_Base = user_sat_base
                                Lightness_Base = user_lig_base

                                Base = [Hue_Base[s], Saturation_Base[s], Lightness_Base[s]]

                                Control_Range = [40, 20, 20]

                                Coff = [1, 1, 1]

                                H = Base[0] + int((p % 16)/8* Control_Range[0]) * Coff[0]
                                Hue = H if H < 360 else 360 - H
                                Saturation = (Base[1] + abs(Ycc[0]) * max(pitch_mag[p] / 50, 2) * Control_Range[1]) * Coff[1]
                                Lightness = (Base[2] + abs(Ycc[1]) * max(pitch_mag[p] / 50, 2) * Control_Range[2]) * Coff[2]

                                color = f"hsl({Hue},{Saturation}%,{Lightness}%)"
                                size = min( max(pitch_mag[p]/2.5 ,1), 50)
                                note_pic.append({
                                    "id": ID,
                                    "pitch": p,
                                    "x": x,
                                    "y": y,
                                    "size": size,
                                    "color": color,
                                    "opacity": 1,
                                    "emotion": Emotion,
                                    "arousal": Ycc[0],
                                    "valence": Ycc[1],
                                    "life": 10
                                })
                                pitch_id[p] = ID
                                pitch_record[p] = current_time
                                ID += 1

                        json_data = json.dumps(note_pic)
                        send_to_clients(f"data: {json_data}\n\n")
                        pitch_active = np.zeros(128)
                        pitch_mag = np.zeros(128)
                        d_time = time.time() - (T_start + count_T)
                        if d_time < 0.1:
                            time.sleep(0.1 - d_time)
                time_record += pitch_times[-1]
                with lock:
                    T_display.append(time.time())
            else:
                time.sleep(0.1)  # 等待更多数据到达



    def Simulator():
        ssr = 4410

        for t in range(len(audio) // ssr):  # 检查 notes_midi 是否为空
            if stop_event.is_set():
                if not processing_event.is_set():
                    stop_event.clear()
                simulator.clear()
                return {"status": "Stopped"}, 200
            start_time = time.time()
            data = audio[t * ssr:t * ssr + ssr].tolist()
            with lock:
                l = len(long_term_store)
                long_term_store.extend(data)
                T_sending.append(time.time())
            D_time = time.time() - start_time
            if D_time < 0.1:
                time.sleep(0.1 - D_time)

    @app.route('/stop', methods=['GET', 'POST'])
    @login_required
    def stop():
        long_term_store.clear()
        stop_event.set()
        processing_event.set()
        mer_event.set()
        with lock:
            feedback_value['value'] = request.get_json().get('value')
            print('feedback',feedback_value['value'])
        while True:
            if not processing_event.is_set():
                print("Stop event set")
                min_length = min(len(T_sending), len(T_receiving), len(T_display))
                del T_sending[min_length-1:]
                del T_receiving[:min_length-1:]
                del T_display[:min_length-1:]
                print(T_sending[-1],T_receiving[-1],T_display[-1])
                print(T_receiving[-1]-T_sending[-1],T_display[-1]-T_sending[-1])
                return jsonify({"message": "all_down",
                                        "start_time":T_sending,
                                        "receiving_time":T_receiving,
                                        "display_time":T_display})

    @app.route('/upload', methods=['POST'])
    @login_required
    def upload_file():
        global audio, file_path
        if 'file' not in request.files:
            flash('No file part')
            return render_template('C_index.html', user=current_user)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('C_index.html', user=current_user)
        if file:
            file_content = file.read()
            audio, sr = librosa.load(io.BytesIO(file_content), sr=44100)
            print(audio.shape)
            flash('File successfully uploaded')
            if not processing_event.is_set():
                processing_event.set()
                mer_event.set()
                user_email = current_user.user_email
                user_hue_base = current_user.user_hue_base
                user_sat_base = current_user.user_sat_base
                user_lig_base = current_user.user_lig_base
                threading.Thread(target=process_data, args=(user_email, user_hue_base, user_sat_base, user_lig_base)).start()
                threading.Thread(target=MER, args=()).start()
            if not simulator.is_set():
                simulator.set()
                threading.Thread(target=Simulator).start()
            print("Started")
            return jsonify({'message': "Start"}), 200

    @app.route('/micro_received', methods=['POST'])
    @login_required
    def micro_received():
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        audio_stream = io.BytesIO(audio_data)
        y, sr = librosa.load(audio_stream)
        yy = librosa.resample(y, orig_sr=sr, target_sr=44100)
        audio_memory.extend(yy)
        print(len(audio_memory))
        if len(audio_memory) >= 4410:
            Data = audio_memory[-4410:]
            del audio_memory[-4410:]
            with lock:
                long_term_store.extend(Data)
                T_sending.append(time.time())

            if not processing_event.is_set():
                processing_event.set()
                mer_event.set()
                user_email = current_user.user_email
                user_hue_base = current_user.user_hue_base
                user_sat_base = current_user.user_sat_base
                user_lig_base = current_user.user_lig_base
                threading.Thread(target=process_data, args=(user_email, user_hue_base, user_sat_base, user_lig_base)).start()
                print("process engaged")
                threading.Thread(target=MER, args=()).start()
                print("MER engaged")

        return jsonify({'message': 'Audio data received successfully'})

    return app
