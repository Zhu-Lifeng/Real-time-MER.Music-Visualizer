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
    outputting = []
    T_sending = []
    T_receiving = []
    T_display = []
    feedback_value = ""
    processing_event = threading.Event()
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
            'user_hue_base': [int(request.form.get('lvla_H')),int(request.form.get('hvla_H')),int(request.form.get('lvha_H')),
                              int(request.form.get('hvha_H')),int(request.form.get('nn_H'))],
            'user_sat_base': [int(request.form.get('lvla_S')), int(request.form.get('hvla_S')), int(request.form.get('lvha_S')),
                              int(request.form.get('hvha_S')), int(request.form.get('nn_S'))],
            'user_lig_base': [int(request.form.get('lvla_L')), int(request.form.get('hvla_L')), int(request.form.get('lvha_L')),
                              int(request.form.get('hvha_L')), int(request.form.get('nn_L'))]
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
                        # 注意: 心跳信号的内容需要符合客户端处理逻辑
                        yield 'data: {}\n\n'  # 发送空数据包来保持连接
            except GeneratorExit:
                # 当客户端断开连接时，清理操作
                clients.remove(q)

        return Response(gen(), mimetype='text/event-stream')

    def process_data(user_email,user_hue_base,user_sat_base,user_lig_base):
        global feedback_value
        print("Process started")
        T_start = time.time()
        count_T = 0
        pitch_record = np.zeros(128)
        pitch_mag = np.zeros(128)
        pitch_id = np.zeros(128)
        time_record = 0.0000001
        middle = [200, 200]
        X_recording = []
        Y_recording = []
        emotion_source = []
        EC=0
        ID = 1
        a = 0
        pitch_active = np.zeros(128)
        note_pic = []
        note_pic_dead = []
        angle = np.linspace(0, 2 * math.pi, 360)
        radius = np.linspace(0, 200, 128)
        Yc = torch.tensor([0,0])
        while True:
            if stop_event.is_set():
                if feedback_value != "cancel":
                    print("sss",feedback_value)
                    T = time.time()
                    print('Stop time:',T)  # 确保这个打印语句可以执行
                    print("Stop event is set. Performing cleanup and exiting.")
                    X = torch.stack(X_recording, dim=0)
                    Y = torch.stack(Y_recording, dim=0)
                    E = feedback_value
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
                    T_sending_F = T_sending[:min_length]
                    T_receiving_F = T_receiving[:min_length]
                    T_display_F = T_display[:min_length]
                    csv_stream = io.BytesIO()
                    text_stream = io.TextIOWrapper(csv_stream, encoding='utf-8', newline='')
                    # 创建 CSV 写入器
                    csv_writer = csv.writer(text_stream)
                    # 写入 CSV 文件，行对行写入
                    for i in range(min_length):
                        csv_writer.writerow([T_sending[i], T_receiving[i], T_display[i]])

                    # 刷新 TextIOWrapper 缓存并重置 BytesIO 流的位置
                    text_stream.flush()
                    csv_stream.seek(0)
                    blob_path_TD = f"{user_email}/{T}/Time_recording.csv"
                    blob_TD = bucket.blob(blob_path_TD)
                    blob_TD.upload_from_file(csv_stream, content_type='text/csv')

                    print("uploaded")  # 确保这个打印语句可以执行
                    print(len(T_sending),len(T_receiving),len(T_display))
                stop_event.clear()
                processing_event.clear()
                return {"status": "Stopped"}, 200
            with lock:
                l = len(long_term_store)
            if l >= 4410:
                #heart_beat = time.time()
                with lock:
                    short_term_store = long_term_store[:4410]
                    del long_term_store[:4410]
                    T_receiving.append(time.time())
                    emotion_source.extend(short_term_store)
                    print("emotion_source", len(emotion_source))
                    EC+=1
                    print("cut", l, len(long_term_store))
                pitches, magnitudes = librosa.piptrack(y=np.array(short_term_store), sr=44100, hop_length=441,threshold=0.5)
                pitch_times = librosa.times_like(pitches, sr=44100, hop_length=441)
                if EC==10:
                    print("MER")
                    EC = 0
                    if len(emotion_source) >= 441000:
                        ess = np.array(emotion_source)
                    else:
                        ess=np.zeros(441000)
                        ess[-len(emotion_source):] = np.array(emotion_source)
                    S = librosa.feature.melspectrogram(y=ess, sr=44100, n_mels=26, hop_length=441)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    zcr = librosa.feature.zero_crossing_rate(ess, hop_length=441)
                    mfccs = librosa.feature.mfcc(y=ess, sr=44100, n_mfcc=26, hop_length=441)
                    F = torch.tensor(np.stack([np.vstack((S_dB, zcr, mfccs))], axis=0))
                    F = [F[ :, :, i * 100:i * 100 + 100] for i in range(10)]
                    F = torch.tensor(np.stack(F, axis=1))
                    X_recording.append(F)
                    Y = model(F.float())
                    Y_recording.append(Y)
                    Yc = Y[0, :]
                    print(Yc)
                    if len(emotion_source) >= 441000:
                        with lock:
                            del emotion_source[44100:]
                for j in range(pitches.shape[1]):
                    current_time = pitch_times[j] + time_record
                    for i in range(pitches.shape[0]):
                        if magnitudes[i, j] > 0:
                            midi_note = int(librosa.hz_to_midi(pitches[i, j]))
                            pitch_active[midi_note] = 1
                            if pitch_mag[midi_note] < magnitudes[i, j]:
                                pitch_mag[midi_note] = magnitudes[i, j]

                    # 更新所有音符圆的信息
                    if j % pitches.shape[1] == 0:  # per 0.1s
                        count_T += 0.1
                        for p in range(128):
                            if pitch_active[p] == 0 and pitch_record[p] != 0:  # 需要消除的圆（已结束的音）
                                note_pic_dead += [item for item in note_pic if item["id"] == pitch_id[p]]
                                note_pic = [item for item in note_pic if (item["id"] != pitch_id[p]) & item["life"]>=0]
                                pitch_id[p] = 0
                                pitch_record[p] = 0

                        for element in note_pic:
                            element["size"] += 0.5
                            element["opacity"] -= 0.02
                            if element["opacity"] < 0:
                                element["opacity"] = 0
                            element["life"]-=0



                        for p in range(128):
                            if pitch_active[p] != 0 and pitch_record[p] == 0 and current_time - pitch_record[p] > 0.05:
                                # 新产生的圆（新出现的音）
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
                                s = 0
                                if ((Yc[0] < 0) & (Yc[1] < 0)) & ((Yc[0] < -0.1) | (Yc[1] < -0.1)):  # Sad / Bored
                                    s = 0
                                elif ((Yc[0] < 0) & (Yc[1] > 0)) & ((Yc[0] < -0.1) | (Yc[1] > 0.1)):  # Content / Relaxed
                                    s = 1
                                elif ((Yc[0] > 0) & (Yc[1] < 0)) & ((Yc[0] > 0.1) | (Yc[1] < -0.1)):  # Angry / Frustrated
                                    s = 2
                                elif ((Yc[0] > 0) & (Yc[1] > 0)) & ((Yc[0] > 0.1) | (Yc[1] > 0.1)):  # Excited / Happy
                                    s = 3
                                else:
                                    s = 4
                                Emotion = "Sad / Bored" if s == 0 else "Content / Relaxed" if s == 1 else "Angry / Frustrated" if s == 2 else "Excited / Happy"\
                                    if s == 3 else "neutral"

                                Hue_Base = user_hue_base
                                Saturation_Base = user_sat_base
                                Lightness_Base = user_lig_base

                                Base = [Hue_Base[s], Saturation_Base[s], Lightness_Base[s]]

                                Control_Range = [20, 20, 20]

                                Coff = [1, 1, 1]

                                H = Base[0] + int((p % 16) * Control_Range[0]) * Coff[0]
                                Hue =  H if H<360 else 360-H
                                Saturation = (Base[1] + abs(Yc[0])* max(pitch_mag[p]/50,2) * Control_Range[1]) * Coff[1]
                                Lightness = (Base[2] + abs(Yc[1]) * max(pitch_mag[p]/50,2) * Control_Range[2]) * Coff[2]

                                color = (
                                    f"hsl({Hue},"
                                    f"{Saturation}%,"
                                    f"{Lightness}%)")
                                size = min(pitch_mag[p] / 10, 50)  # 初始圆的尺寸
                                note_pic.append({
                                    "id": ID,
                                    "pitch": p,
                                    "x": x,
                                    "y": y,
                                    "size": size,
                                    "color": color,
                                    "opacity": 1,
                                    "emotion": Emotion,
                                    "arousal": Yc[0].float().item(),
                                    "valence": Yc[1].float().item(),
                                    "life": 20
                                })
                                pitch_id[p] = ID
                                pitch_record[p] = current_time
                                ID += 1

                        json_data = json.dumps(note_pic)
                        send_to_clients(f"data: {json_data}\n\n")
                        pitch_active = np.zeros(128)
                        pitch_mag = np.zeros(128)
                        d_time = time.time() - (T_start+count_T)
                        if d_time < 0.1:
                            time.sleep(0.1 - d_time)
                time_record += pitch_times[-1]
                T_display.append(time.time())
            else:
                time.sleep(0.1)  # 等待更多数据到达
                T = time.time()
                print("wait",T)


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
                print("append",l,len(long_term_store))
            D_time = time.time() - start_time
            if D_time < 0.1:
                time.sleep(0.1 - D_time)


    @app.route('/stop',methods =['GET','POST'])
    @login_required
    def stop():
        global feedback_value
        long_term_store.clear()
        stop_event.set()
        processing_event.set()
        with lock:
            feedback_value = request.args.get('value')

        print("Stop event set")  # 添加打印以确认事件被设置
        return render_template('C_index.html', user=current_user)

    @app.route('/upload', methods=['POST'])
    @login_required
    def upload_file():
        global audio, sr, file_path
        if 'file' not in request.files:
            flash('No file part')
            return render_template('C_index.html', user=current_user)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('C_index.html', user=current_user)
        if file:
            #filename =file.filename
            #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(file_path)
            file_content = file.read()
            audio, sr = librosa.load(io.BytesIO(file_content), sr=44100)
            print(audio.shape)
            flash('File successfully uploaded')
            print(processing_event.is_set())
            print(simulator.is_set())
            if not processing_event.is_set():
                processing_event.set()  # 标记处理事件为已设置
                user_email = current_user.user_email
                user_hue_base = current_user.user_hue_base
                user_sat_base = current_user.user_sat_base
                user_lig_base = current_user.user_lig_base
                threading.Thread(target=process_data,args=(user_email,user_hue_base,user_sat_base,user_lig_base)).start()
            if not simulator.is_set():
                simulator.set()
                threading.Thread(target=Simulator).start()
            print("Started")
            return jsonify({'message' : "Start"}),200
            #return render_template('C_index.html', user=current_user)


    @app.route('/micro_received',methods=['POST'])
    @login_required
    def micro_received():
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        audio_stream = io.BytesIO(audio_data)
        y, sr = librosa.load(audio_stream)
        Data = librosa.resample(y, orig_sr=sr, target_sr=44100)
        with lock:
            l = len(long_term_store)
            long_term_store.extend(Data)
            print("append",l,len(long_term_store))
            T_sending.append(time.time())
        if not processing_event.is_set():
            processing_event.set()  # 标记处理事件为已设置
            user_email = current_user.user_email
            user_hue_base = current_user.user_hue_base
            user_sat_base = current_user.user_sat_base
            user_lig_base = current_user.user_lig_base
            threading.Thread(target=process_data, args=(user_email,user_hue_base,user_sat_base,user_lig_base)).start()

        return jsonify({'message': 'Audio data received successfully'})

    return app
