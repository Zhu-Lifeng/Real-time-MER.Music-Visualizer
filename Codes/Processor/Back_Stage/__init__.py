from flask import Flask, render_template, request, Response, jsonify
import numpy as np
import requests
import threading
import librosa
import mido
import json
import time
import math
def Processor_Creation():
    global long_term_store

    app = Flask(__name__)
    long_term_store = []

    processing_event = threading.Event()  # 创建一个事件对象


    @app.route('/')
    def index():
        return render_template('P_index.html')

    @app.route('/stream')
    def stream():
        def event_stream():
            old_length = len(long_term_store)
            while True:
                if len(long_term_store) > old_length:
                    new_data = len(long_term_store)
                    old_length = len(long_term_store)
                    yield f"data: {json.dumps(new_data)}\n\n"
                #time.sleep(1)

        return Response(event_stream(), mimetype='text/event-stream')

    def process_data(target_ip, port):
        global long_term_store
        pitch_record = np.zeros(128)
        time_record = 0.01
        middle = [250, 250]
        id = 0
        a=0

        angle = np.linspace(0,2*math.pi,360)
        radius = np.linspace(0, 250,128)
        url = f'http://{target_ip}:{port}/audio_Msg_received'
        while True:
            print(len(long_term_store))
            if len(long_term_store) >= 4410:
                print("1")
                responses = []
                note_pattern = []
                note_pic=[]
                pitch_active = np.zeros(128)
                pitch_mag = np.zeros(128)
                pitch_tol = np.zeros(128)
                pitch_id = np.zeros(128)
                short_term_store = long_term_store[0:4410]
                long_term_store = long_term_store[4410:]
                print("2")
                pitches, magnitudes = librosa.piptrack(y=np.array(short_term_store), sr=44100, hop_length=512, threshold=0.1)
                print("3")
                pitch_times = librosa.times_like(pitches, sr=44100, hop_length=512)
                print("4")

                for j in range(pitches.shape[1]):
                    current_time = pitch_times[j] + time_record
                    pitch_active = np.zeros(128)
                    for i in range(pitches.shape[0]):
                        if magnitudes[i, j] > 0:
                            midi_note = int(librosa.hz_to_midi(pitches[i, j]))
                            pitch_active[midi_note] = 1
                            pitch_mag[midi_note] += magnitudes[i, j]

                    if j//441 == 0:
                        #更新所有音符圆的信息
                        for p in range(128):
                            if pitch_active[p] == 0 and pitch_record[p] != 0: #需要消除的圆（已结束的音）
                                if current_time - pitch_record[p] > 0.05:
                                    #note_pattern.append([p, f"{pitch_record[p]:.2f}", f"{(current_time - pitch_record[p]):.2f}"])
                                    note_pic = [item for item in note_pic if item["id"] != pitch_id[p]]
                                pitch_record[p] = 0

                        for element in note_pic:
                            element["size"] += 1

                        for p in range(128):
                            if pitch_active[p] != 0 and pitch_record[p] == 0: # 新产生的圆（新出现的音）
                                if a < 360:
                                    angle_N = angle[a]
                                    a+=1
                                else:
                                    a -= 360
                                    angle_N = angle[a]
                                    a+=1
                                radius_N = radius[p]
                                x = middle[0] + radius_N * math.cos(angle_N)
                                y = middle[1] + radius_N * math.sin(angle_N)
                                color =(f"rgb({int(min(pitch_mag[p]/350,1) * 255)},{int(min(pitch_mag[p]/350,1) * 255)}, {int(min(pitch_mag[p]/350,1) * 255)})")
                                size = 10#初始圆的尺寸
                                note_pic.append({
                                        "id": id,
                                        "pitch": p,
                                        "x": x,
                                        "y": y,
                                        "size": size,
                                        "color": color,
                                    })
                                pitch_id[p] = id
                                pitch_record[p] = current_time
                                id += 1

                    time_record += pitch_times[-1]
                if note_pic != []:
                    data = note_pic
                    response = requests.post(url, json=data)
                    print("Sent")
                    responses.append(response.status_code)
            else:
                time.sleep(0.25)  # 等待更多数据到达


    @app.route('/audio_fragment_receive', methods=['POST'])
    def receive_data():
        global pitch_record, long_term_store,time_record
        data = request.get_json()
        long_term_store += data

        return {"status": "Data received"}, 200


    @app.route('/audio_Msg_send', methods=['GET', 'POST'])
    def send_Msg():
        if request.method == 'POST':
            if not processing_event.is_set():
                processing_event.set()  # 标记处理事件为已设置
                #target_ip = request.form.get('target_ip')
                #port = request.form.get('port')
                target_ip = '127.0.0.1'
                port='8002'
                threading.Thread(target=process_data, args=(target_ip, port)).start()

            return {"status": "Start working"}, 200




    return app
