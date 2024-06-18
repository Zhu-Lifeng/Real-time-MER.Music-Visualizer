import queue
from flask import Flask, render_template, request, Response, jsonify
import numpy as np
import threading
import librosa
import mido
import json
import time
import math


def Processor_Creation():
    app = Flask(__name__)
    long_term_store = []
    clients = []
    outputting = []
    processing_event = threading.Event()  # 创建一个事件对象
    stop_event = threading.Event()
    lock = threading.Lock()

    @app.route('/')
    def index():
        return render_template('P_index.html')

    @app.route('/audio_fragment_receive', methods=['POST'])
    def receive_data():
        data = request.get_json()
        with lock:
            long_term_store.extend(data)
            print("received", len(long_term_store))
        return {"status": "Data received"}, 200

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
    def register_client():
        def gen():
            q = queue.Queue()
            clients.append(q)
            P_time = 0
            try:
                while True:
                    try:
                        data = q.get(timeout=10)  # 设置超时以避免长时间阻塞
                        D_time = time.time() - P_time
                        if D_time < 0.1:
                            time.sleep(0.1-D_time)
                        yield data
                        P_time = time.time()
                    except queue.Empty:
                        # 如果超时没有数据，发送一个保持连接的心跳信号
                        # 注意: 心跳信号的内容需要符合客户端处理逻辑
                        yield 'data: {}\n\n'  # 发送空数据包来保持连接
            except GeneratorExit:
                # 当客户端断开连接时，清理操作
                clients.remove(q)

        return Response(gen(), mimetype='text/event-stream')

    def process_data():
        pitch_record = np.zeros(128)
        pitch_mag = np.zeros(128)
        pitch_id = np.zeros(128)
        time_record = 0.00001
        middle = [250, 250]
        ID = 1
        a = 0
        pitch_active = np.zeros(128)
        note_pic = []
        angle = np.linspace(0, 2 * math.pi, 360)
        radius = np.linspace(0, 250, 128)
        while True:
            if stop_event.is_set():
                stop_event.clear()
                break
            with lock:
                l = len(long_term_store)
                print("working", l)
            if l >= 441000:
                with lock:
                    short_term_store = long_term_store[:441000]
                    del long_term_store[:441000]
                    print("cut", len(long_term_store))

                pitches, magnitudes = librosa.piptrack(y=np.array(short_term_store), sr=44100, hop_length=1024,
                                                       threshold=0.1)

                pitch_times = librosa.times_like(pitches, sr=44100, hop_length=1024)
                for j in range(pitches.shape[1]):
                    start_time = time.time()
                    current_time = pitch_times[j] + time_record

                    for i in range(pitches.shape[0]):
                        if magnitudes[i, j] > 0:
                            midi_note = int(librosa.hz_to_midi(pitches[i, j]))
                            pitch_active[midi_note] = 1
                            pitch_mag[midi_note] += magnitudes[i, j]

                    if j//4410 == 0:
                        # 更新所有音符圆的信息
                        for p in range(128):
                            if pitch_active[p] == 0 and pitch_record[p] != 0:  # 需要消除的圆（已结束的音）
                                note_pic = [item for item in note_pic if item["id"] != pitch_id[p]]
                                pitch_id[p] = 0
                                pitch_record[p] = 0

                        for element in note_pic:
                            element["size"] += 0.1

                        for p in range(128):
                            if pitch_active[p] != 0 and pitch_record[p] == 0 and current_time-pitch_record[p] > 0.05:  # 新产生的圆（新出现的音）
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
                                color = (
                                        f"rgb({int((1-min(pitch_mag[p] / 200, 1)) * 255*0.5)},"
                                        f"{int((min(pitch_mag[p] / 200, 1)) * 255*0.75)},"
                                        f"{int((1-min(pitch_mag[p] / 200, 1)) * 255)*0.6})")
                                size = 5  # 初始圆的尺寸
                                note_pic.append({
                                        "id": ID,
                                        "pitch": p,
                                        "x": x,
                                        "y": y,
                                        "size": size,
                                        "color": color,
                                    })
                                pitch_id[p] = ID
                                pitch_record[p] = current_time
                                ID += 1
                        print("pic:",len(note_pic))
                        json_data = json.dumps(note_pic)
                        send_to_clients(f"data: {json_data}\n\n")
                        pitch_active = np.zeros(128)
                        pitch_mag = np.zeros(128)
                    d_time = time.time()-start_time
                    if d_time < 0.01:
                        time.sleep(0.01-d_time)
                time_record += pitch_times[-1]
            else:
                print("alive", l)
                time.sleep(0.5)  # 等待更多数据到达

    @app.route('/audio_Msg_send', methods=['GET', 'POST'])
    def send_Msg():
        if request.method == 'POST':
            if not processing_event.is_set():
                processing_event.set()  # 标记处理事件为已设置
                # target_ip = request.form.get('target_ip')
                # port = request.form.get('port')
                # target_ip = '127.0.0.1'
                # port='8002'
                threading.Thread(target=process_data).start()
                print("Started")

            return render_template('C_index.html')

    @app.route('/stop',methods =['POST'])
    def stop():
        long_term_store.clear()
        stop_event.set()
        return render_template('P_index.html')
    return app
