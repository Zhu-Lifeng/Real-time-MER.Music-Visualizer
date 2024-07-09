from Back_Stage import Processor_Creation
import ssl
app=Processor_Creation()

if __name__ == '__main__':
  with app.app_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('ssl_cert.crt', 'ssl_private.key', '1998')

    app.run(debug=True,  ssl_context=context, host='0.0.0.0',port=8080) 
