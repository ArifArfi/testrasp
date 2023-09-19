import base64
# import logging
# LogUtil external file
# from LogUtil import LogUtil
import traceback
import paho.mqtt.client as mqtt_client
import paho.mqtt.publish as publish
import time
from datetime import datetime, date
# import cv2

import numpy as np  # linear algebra
# from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from PIL import Image

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC

import io
import queue
import threading
from time import sleep
from tensorflow.keras.preprocessing import image
import json

import requests
from bs4 import BeautifulSoup

base_url = "http://103.161.184.75:80/"

# LogUtil.setup_logging()
# logger = logging.getLogger(__name__)
counter: int = 0
q = queue.Queue()


# load the facenet model
facenet_model = load_model('model-cnn-facerecognition-labelid.tflite')
# logger.info('Loaded Model')


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        # logger.info("Connected to broker")
        global Connected
        Connected = True
    else:
        # logger.error("Connection failed")
        pass


def on_message(client, userdata, message):
    try:
        # logger.info("topic {}".format(message.topic))
        # logger.info("message {}".format(message.payload.decode()))

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        fmt_hex = format(int(time.mktime(time.strptime(now, '%Y-%m-%d %H:%M:%S'))) - time.timezone, 'x').upper()
        global counter
        counter += 1
        name_file = "{}".format(f'{counter:06}')  # zero padding
        # logger.info("current time: {} {} {}".format(now, fmt_hex, name_file))

        q.put(message.payload)

    except Exception as e:
        traceback.print_exc()
        # logger.error(e)

def get_labels_from_API():

    permintaan = base_url + "api/pegawai"
    response = requests.get( permintaan )
    raw = response.json()
    labels = []

    try:
        if raw['status'] is True:
            for key in raw['data']:
                labels.append(str(key["id"]) + ";" + key["nama"])
    except Exception as e:
        print(e)
        pass

    return labels

def predict_image(name):
    labels = get_labels_from_API()
    if not labels:
        # logger.error("No labels found ")
        return

    # logger.info('Thread %s: starting', name)
    global running
    running = True
    while running:
        if not q.empty():
            # logger.info('Size of queue: %d', q.qsize())
            msg_base64 = q.get()

            try:
                dt_payload_split = str(msg_base64, 'utf-8').split("_")
                base64_citra = dt_payload_split[1].encode("utf-8")
                device_id = dt_payload_split[0]
                citra = Image.open(io.BytesIO(base64.decodebytes(base64_citra)))
                X = image.img_to_array(citra)
                X = np.expand_dims(X, axis=0)
                images = np.vstack([X])
                val = facenet_model.predict(images)
                listHasil = []
                for i in val[0]:
                    listHasil.append(i)
                getMax = np.max(val[0])
                getindex = listHasil.index(getMax)
                # print(f"Terprediksi label {labels[getindex]} dengan akurasi {getMax}")
                title = '%s (%.3f)' % (labels[getindex], getMax)
                # logger.info(title)
                client.publish(topic=out_topic_face, payload=labels[getindex])


                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Menggunakan datetime untuk mendapatkan waktu terkini
                data_json = {
                    "data" : str(base64_citra, 'utf-8'),
                    "device_id" : device_id,
                    "waktu" : now,
                    "label" : labels[getindex],
                    "persentase" : getMax.item()
                }
                json_result = json.dumps(data_json, indent=4)
                print(json_result)
                publish.single("face/recognition", payload=json_result)

                # Send data (POST)
                destinasi_request = "api/do_attendance"

                # split label
                pemisah = (labels[getindex]).split(";")
                id_nip = pemisah[0]

                link_API = base_url + destinasi_request
                r = requests.get(base_url + "scrapper-csrf")  # url is create form
                soup = BeautifulSoup(r.text, 'html.parser')
                token = ""
                for link in soup.find_all('input'):
                    token = link['value']

                parameter = {
                    'device_id' : device_id,
                    'pegawai_id': id_nip,
                    'time': now,
                    '_token': token
                }
                jar = requests.cookies.RequestsCookieJar()
                jar.set('laravel_session', r.cookies['laravel_session'])

                r = requests.post(link_API, data=parameter, cookies=jar)
                print(r.text)

                # print(type(msg_base64))
                # time.sleep(3)
            except Exception as e:
                print(e)
                pass

        sleep(0.1)
    # logger.info('Thread %s: finished', name)

# mqtt address
broker_address = "127.0.0.1"
port = 1883

Connected = False
mqtt_sub = "camera/photo"

out_topic_face = "camera/photo"

client = mqtt_client.Client()
client.on_connect = on_connect
client.on_message = on_message
client.subscribe("face/recognition")

thread = threading.Thread(target=predict_image, args=('Robot',))
thread.start()

try:
    # logger.info("Connecting to {} {}".format(broker_address, port))
    client.connect(broker_address, port)
    client.loop_start()
except Exception as e:
    traceback.print_exc()
    # logger.error(e)

while not Connected:
    time.sleep(0.1)
client.subscribe(mqtt_sub)
# logger.info("Connected {} {}".format(broker_address, port))

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # logger.warning("exiting")
    global running
    running = False
    client.disconnect()
    client.loop_stop()