import cv2 as cv
import numpy as np
import torch #lib que carrega o YOLO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cam = cv.VideoCapture(0)

def detection():
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        results = model(frame) #processa e detecta os objetos da webcam
        pred = results.pandas().xyxy[0]
        
        for index, row in pred.iterrows():
            box = [int(x) for x in row[['xmin', 'ymin', 'xmax', 'ymax']]]  # retorna as coordenada x e y iniciais da caixa delimitadora, assim como os x e y finais

            confidence = row['confidence'] * 100 # confiança (em porcentagem) da detecção 
            confidence = round(confidence, 0) # retira as casas decimais
            class_id = int(row['class']) # retorna o id da classe classe detectada 

            if class_id == 0:
                cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv.putText(frame, f'pessoa {confidence}%', (box[0], box[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
            # buffer = cv.imencode('.jpg', frame)
            # data_encode = np.array(buffer) 
            # frame = data_encode.tobytes()
            # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cv.imshow('detection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

detection()