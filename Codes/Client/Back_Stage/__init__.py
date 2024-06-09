from flask import Flask, render_template, Response, request
import json
import time

def Client_Creation():
    app = Flask(__name__)
    data_store = []

    @app.route('/')
    def index():
        return render_template('C_index.html')

    @app.route('/audio_Msg_received', methods=['POST'])
    def receive_data():
        data = request.get_json()
        data_store.append(data)
        return {"status": "Data received"}, 200

    @app.route('/stream')
    def stream():
        def event_stream():
            old_length = len(data_store)
            while True:
                if len(data_store) > old_length:
                    new_data = data_store[-1]
                    old_length = len(data_store)
                    yield f"data: {json.dumps(new_data)}\n\n"
                #time.sleep(1)

        return Response(event_stream(), mimetype='text/event-stream')

    return app
