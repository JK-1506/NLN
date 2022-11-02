# import thư viện
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import pyautogui
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Lấy các kích thước của khung và sau đó tạo một đốm màu từ nó
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

    # Thông qua đốm màu qua mạng và phát hiện khuôn mặt
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

    # Khởi tạo danh sách các khuôn mặt, vị trí tương ứng của chúng và danh sách các dự đoán từ mạng mặt nạ 
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

    # Vòng lặp các phát hiện
	# loop over the detections
	for i in range(0, detections.shape[2]):
        # Độ tin cậy liên quan đến phát hiện 
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

        # Lọc ra các phát hiện yếu bằng cách đảm bảo độ tin cậy lớn hơn độ tin cậy tối thiểu
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
            # Tính toán tọa độ (x, y) của hộp giới hạn cho đối tượng
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
               
            # Đảm bảo các giới hạn hộp nằm trong kích thước của khung
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Trích xuất ROI của khuôn mặt, chuyển đổi nó từ thứ tự kênh BGR sang RGB, thay đổi kích thước thành 224x224 và xử lý trước
			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

            # Thêm các hộp mặt và khung vào danh sách tương ứng của chúng
			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

    # Chỉ đưa ra dự đoán nếu ít nhất một khuôn mặt được phát hiện
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
        # Để suy luận nhanh hơn, chúng tôi sẽ đưa ra dự đoán hàng loạt trên * tất cả * mặt cùng một lúc thay vì dự đoán từng cái một trong vòng lặp `for` ở trên
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

    # Trả về 2 bộ vị trí khuôn mặt và các vị trí tương ứng của chúng
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# Tải mô hình máy dò khuôn mặt được tuần tự hóa của chúng tôi từ đĩa
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Tải mô hình phát hiện mặt nạ từ đĩa
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Khởi tạo đầu vào video
# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Lặp lại các khung hình từ luồng video
# loop over the frames from the video stream
while True:
    # Lấy khung hình từ luồng video theo chuỗi và thay đổi kích thước để có chiều rộng tối đa là 400 pixel
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=700)

    # Phát hiện các khuôn mặt trong khung hình và xác định xem họ có đeo mặt nạ hay không
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Lặp lại các vị trí khuôn mặt được phát hiện và các vị trí tương ứng của chúng
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
        # Giải nén hộp giới hạn và dự đoán
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
        
        # Xác định nhãn lớp và màu sắc mà chúng ta sẽ sử dụng để vẽ hộp giới hạn và văn bản
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #if label == "Mask" else pyautogui.screenshot("screenshot.png")
        
        # Bao gồm xác suất trong nhãn
		# include the probability in the label
		label = "{}".format(label)#, max(mask, withoutMask) * 100) #: {:.2f}%

        # Hiển thị nhãn và hình chữ nhật hộp giới hạn trên khung đầu ra
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Hiển thị khung đâu ra
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
    
    # nếu phím `q` được nhấn, hãy ngắt khỏi vòng lặp
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Dọn dẹp
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()