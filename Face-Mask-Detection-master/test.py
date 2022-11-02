import imutils
import pyautogui
import cv2

# Another Type
pyautogui.screenshot("screenshot.png")

# we can then load our screenshot from disk in OpenCV format
image = cv2.imread("screenshot.png")
cv2.imshow("vinasupport.com", imutils.resize(image, width=600))
cv2.waitKey(0)