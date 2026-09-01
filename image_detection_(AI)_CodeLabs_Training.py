import cv2
import numpy as np
import pytesseract

#===================================
#OSR Path
#===================================

#place tesseract on windows
pytesseract.pytesseract.tesseract_cmd = r"c:\Program Files\Tesseract-OCR\tesseract.exe"

#reading image
image = cv2.imread("text_test.png")
if image is None:
    print("Error: Image not found")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Adaptive thresholding
threshold = cv2.adaptiveThreshold(
    gray,
    255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,
11,
    5
)


#convert image to Grayscale
text = pytesseract.image_to_string(gray)
print("Extracted Text:")
print(text)


#Show original image to know if processing done
cv2.imshow("Original Image", image)
cv2.imshow("Processed Image", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()


#===================================
#Opject Detection Path
#===================================


#load the pre-trained MobileNet-SSD
net =cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")


#object classes
classes = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

#read the object image
object_image = cv2.imread("image_test.jpg")
if object_image is None:
    print("Error: Object Image not found")
    exit()

#get image dimensions and convert it into a blob
(h, w) = object_image.shape[:2]
blob = cv2.dnn.blobFromImage(object_image, 0.007843, (300, 300), 127.5)
#give the blob to the neural network
net.setInput(blob)
#run the model
detections = net.forward()
#check every detection
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    #accept only detections with 80% confidence or higher
    if confidence >= 0.80:
        class_id = int(detections[0, 0, i, 1])
        label = classes[class_id]
        #get bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        #draw the bounding box
        cv2.rectangle(object_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        #creat label with confidence
        text = f"{label}: {confidence * 100:.1f}%"
        #put label on the image
        cv2.putText(object_image,text,(startX, max(startY- 10, 15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    #show result
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", 800, 600)
    cv2.imshow("Object Detection", object_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()