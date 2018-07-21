import cv2
from PIL import Image
import numpy as np
import time

# image path for any filter you want to use
maskPath = "C:/Users/laptop care/Desktop/python/Snapchat-Filters-PNG-Image-180x180.png"
# haarcascade path
cascPath = "haarcascade_frontalface_default.xml"


# cascade classifier object 
faceCascade = cv2.CascadeClassifier(cascPath)
#for eye detection of person
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open mask as PIL image
mask = Image.open(maskPath)

def snapchat_filter(image):
	"""
	function to add any additional filter like snapchat to input image
	"""

	# convert input image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in grayscale image
	faces = faceCascade.detectMultiScale(gray, 1.15)

	# convert cv2 image to PIL image
	background = Image.fromarray(image)

	for (x,y,w,h) in faces:
		# resize filter image
		resized_mask = mask.resize((w,h), Image.ANTIALIAS)
		# define offset for filter image
		offset = (x-10,y-10)
		# paste filter image on background
		background.paste(resized_mask, offset, mask=resized_mask)

	# return background as cv2 image
	return np.asarray(background)

# VideoCapture object
cap = cv2.VideoCapture(cv2.CAP_ANY)

while True:
	# read return value and frame
	ret, frame = cap.read()

	if ret == True:
		# show frame with snapchat filter
		cv2.imshow('Live', snapchat_filter(frame))

		# check if esc key is pressed
		if cv2.waitKey(1) == 'q':
			break

# release cam
cap.release()
# destroy all open opencv windows
cv2.destroyAllWindows()
