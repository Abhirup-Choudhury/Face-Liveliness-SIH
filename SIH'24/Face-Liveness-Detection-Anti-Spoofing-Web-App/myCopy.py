import cv2
import tensorflow as tf
import numpy as np
import imutils
import onnxruntime as ort

model_path = 'liveness_model.onnx'
le_path = 'label_encoder.pickle'
detector_folder = 'face_detector'
threshold = 0.5

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = "Face-Liveness-Detection-Anti-Spoofing-Web-App\\face_detector\deploy.prototxt"
model_path = "Face-Liveness-Detection-Anti-Spoofing-Web-App\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"

detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
# load the liveness detector model and label encoder from disk
print('[INFO] loading liveliness model...')
session = ort.InferenceSession(
    "Face-Liveness-Detection-Anti-Spoofing-Web-App\liveness_model.onnx")

# class VideoProcessor:


def process_frame(frm):
    # frm = frame.to_ndarray(format="bgr24")

    # iterate over the frames from the video stream
    # while True:
    # grab the frame from the threaded video stream
    # and resize it to have a maximum width of 600 pixels
    height, width = frm.shape[:2]
    aspect = width / height
    new_width = 800
    new_height = int(new_width/aspect)
    frm = cv2.resize(frm, (new_width, new_height))
    # frm = imutils.resize(frm, width=800)

    # grab the frame dimensions and convert it to a blob
    # blob is used to preprocess image to be easy to read for NN
    # basically, it does mean subtraction and scaling
    # (104.0, 177.0, 123.0) is the mean of image in FaceNet
    (h, w) = frm.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network
    # and obtain the detections and predictions
    detector_net.setInput(blob)
    detections = detector_net.forward()

    # iterate over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e. probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > threshold:
            # compute the (x,y) coordinates of the bounding box
            # for the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # expand the bounding box a bit
            # (from experiment, the model works better this way)
            # and ensure that the bounding box does not fall outside of the frame
            startX = max(0, startX-20)
            startY = max(0, startY-20)
            endX = min(w, endX+20)
            endY = min(h, endY+20)

            # extract the face ROI and then preprocess it
            # in the same manner as our training data

            face = frm[startY:endY, startX:endX]  # for liveness detection
            # expand the bounding box so that the model can recog easier
            # some error occur here if my face is out of frame and comeback in the frame
            try:
                # our liveness model expect 32x32 input
                face = cv2.resize(face, (32, 32))
            except:
                break

            # initialize the default name if it doesn't found a face for detected faces
            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            # tf model require batch of data to feed in
            # so if we need only one image at a time, we have to add one more dimension
            # in this case it's the same with [face]
            face = np.expand_dims(face, axis=0)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face})
            # pass the face ROI through the trained liveness detection model
            # to determine if the face is 'real' or 'fake'
            # predict return 2 value for each example (because in the model we have 2 output classes)
            # the first value stores the prob of being real, the second value stores the prob of being fake
            # so argmax will pick the one with highest prob
            # we care only first output (since we have only 1 input)
            # preds = liveness_model.predict(face)[0]
            preds = outputs[0][0]
            j = np.argmax(preds)
            print(j)
            label_name = "null"
            if (j):
                label_name = 'real'
            else:
                label_name = 'fake'
            # draw the label and bounding box on the frame
            label = f'{label_name}: {preds[j]:.4f}'
            # print(f'[INFO] {name}, {label_name}')

            if label_name == 'fake':
                cv2.putText(frm, "Fake Alert!", (startX, endY + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            # cv2.putText(frm, name, (startX, startY - 35),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
            cv2.putText(frm, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 0, 255), 4)
    return frm


cap = cv2.VideoCapture('http://192.168.240.220:4747/video')
while True:
    success, frame = cap.read()
    # frame = cv2.resize(frame, resized)
    pr_frame = process_frame(frame)
    cv2.imshow("Video", pr_frame)

    # if "q" is pressed then quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
