from flask import Flask, render_template, request, jsonify,flash
import time
import pandas as pd
import numpy as np
import json
import requests
import librosa
import os
def Sender_Creation():
    app = Flask(__name__)
    app.secret_key = '19980706'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    #audio, sr = librosa.load('MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav', sr=44100)


    @app.route('/')
    def index():
        return render_template('S_index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        global audio, sr, file_path
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            audio, sr = librosa.load(file_path, sr=44100)
            flash('File successfully uploaded')

            return render_template('S_index.html')


    def secure_filename(filename):
        return filename


    @app.route('/audio_fragment_send', methods=['GET', 'POST'])
    def send_data():
        global audio, sr, file_path

        if request.method == 'POST':
            target_ip = request.form.get('target_ip')
            port = request.form.get('port')
            url = f'http://{target_ip}:{port}/audio_fragment_receive'
            headers = {'Content-Type': 'application/json'}
            responses = []
            ssr = 441
            for t in range(len(audio)//ssr):  # 检查 notes_midi 是否为空
                data = audio[t*ssr:t*ssr+ssr].tolist()
                try:
                    response = requests.post(url, json=data)
                    responses.append((t, response.status_code))
                    print(f'第 {t} 条数据发送成功')
                except requests.exceptions.RequestException as e:
                    print(f"发送数据失败: {e}")
                    responses.append((t, e))
                time.sleep(0.01)  # 根据响应时间需求调整
            #requests.post(url,json=999999)
            return jsonify(responses)  # 返回所有发送状态的汇总

    return app
