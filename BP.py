from flask import Blueprint, jsonify, Response

BP = Blueprint('BP',__name__)

@BP.route('/')
def st():
    return "<h1>Hello, World!</h1>"

@BP.route('/stream')
def stream():
    def generate():
        while True:
            yield

    return Response(generate(), mimetype='text/event-stream')
