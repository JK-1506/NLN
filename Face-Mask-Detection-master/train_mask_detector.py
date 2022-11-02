# import thư viện
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Khởi tạo tốc độ học ban đầu , epochs và kích thước lô.
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\E5-476\Documents\NLN\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Lấy danh sách hình ảnh trong thư mục dữ liệu, khởi tạo danh sách dữ liệu và lớp hình ảnh
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# Thực hiện mã hóa một lần trên các nhãn
# perform one-hot encoding on the labels
lb = LabelBinarizer() #Binarize các nhãn theo kiểu một chọi tất cả.
labels = lb.fit_transform(labels) # gán dãn thành số nguyên
labels = to_categorical(labels) #Chuyển đổi một vectơ lớp (số nguyên) thành ma trận lớp nhị phân.

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)# stratify == phân tầng, mặc định không có
    
# Trình tạo dữ liệu hình ảnh
# construct the training image generator for data augmentation
aug = ImageDataGenerator( #tiền sử lý hình ảnh
	rotation_range=20, #Phạm vi độ cho các phép quay ngẫu nhiên.
	zoom_range=0.15, #Phạm vi thu phóng ngẫu nhiên
	width_shift_range=0.2,#float: phần nhỏ của tổng chiều rộng, nếu <1 hoặc pixel nếu> = 1 
	height_shift_range=0.2, #loat: phần nhỏ của tổng chiều cao, nếu <1 hoặc pixel nếu> = 1.
	shear_range=0.15,#Cường độ cắt
	horizontal_flip=True,#Lật ngẫu nhiên các đầu vào theo chiều ngang.
	fill_mode="nearest")#Các điểm bên ngoài ranh giới của đầu vào được điền theo chế độ đã cho: 'nearest': aaaaaaaa|abcd|dddddddd

# Tải mạng MobileNetV2, đảm bảo các tập hợp lớp FC đầu được tắt
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,#có bao gồm lớp được kết nối đầy đủ ở đầu mạng hay không.
	input_tensor=Input(shape=(224, 224, 3)))# 'imagenet' (đào tạo trước trên ImageNet) hoặc đường dẫn đến tệp weights sẽ được tải. Mặc định thành imagenet.
    #đầu ra của layers.Input()) để sử dụng làm đầu vào hình ảnh cho mô hình. 
# Xây dựng phần đầu mô hình
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output# đầu ra mô hình
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)#Làm giảm giá trị đầu vào dọc theo các kích thước không gian của nó (chiều cao và chiều rộng) bằng cách lấy giá trị trung bình trên một cửa sổ đầu vào (có kích thước được xác định bởi pool_size) cho mỗi kênh của đầu vào
headModel = Flatten(name="flatten")(headModel)#Làm phẳng đầu vào. Không ảnh hưởng đến kích thước lô.
headModel = Dense(128, activation="relu")(headModel)# 
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Đặt mô hình đầu FC lên trên mô hình cơ sở (đây sẽ trở thành mô hình thực tế mà chúng tôi sẽ đào tạo)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Lặp qua tất cả các lớp trong mô hình cơ sở và đóng băng chúng để chúng * không * được cập nhật trong quá trình đào tạo đầu tiên
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False#Cài đặt layer.trainableđể Falsedi chuyển tất cả các trọng lượng của lớp từ có thể huấn luyện sang không thể huấn luyện. Đây được gọi là "đóng băng" lớp: trạng thái của lớp bị đóng băng sẽ không được cập nhật trong quá trình đào tạo (khi đào tạo với fit()hoặc khi đào tạo với bất kỳ vòng lặp tùy chỉnh nào dựa vào đó trainable_weightsđể áp dụng cập nhật gradient).

# Biên dịch mô hình
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])#nh toán số liệu theo hướng chéo giữa các nhãn và dự đoán.,

# Train phần đầu của mạng
# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Dự đoán
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Mỗi hình ảnh cần được gán nhãn và xác suất dự đoán
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1) #Trả về chỉ số của các giá trị lớn nhất dọc theo trục.

# Hiển thị báo cáo phân loại
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# Lưu mô hình
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# Hiển thị loss và accuracy
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")