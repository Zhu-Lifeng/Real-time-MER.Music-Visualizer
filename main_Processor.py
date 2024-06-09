from Processor import Processor_Creation

app=Processor_Creation()

if __name__ == '__main__':
    with app.app_context():

        app.run(debug=True,host='0.0.0.0',port=8001)