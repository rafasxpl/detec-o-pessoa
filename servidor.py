from flask import Flask, Response, render_template
import cv2 as cv
import numpy as np
import torch

app = Flask(__name__)

# Carrega o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cam = cv.VideoCapture(0)

def detection():
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Processa o frame e detecta os objetos
        results = model(frame)
        pred = results.pandas().xyxy[0]
        
        for index, row in pred.iterrows():
            box = [int(x) for x in row[['xmin', 'ymin', 'xmax', 'ymax']]]
            confidence = round(row['confidence'] * 100, 0)
            class_id = int(row['class'])

            if class_id == 0:  # Se for uma pessoa
                cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv.putText(frame, f'Pessoa {confidence}%', (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Codifica o frame em JPEG
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Retorna o frame como parte do fluxo multipart
        
        yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

# Libera a câmera após o término do uso
cam.release()
