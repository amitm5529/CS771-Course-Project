# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR
import numpy as np
import cv2
import pickle

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.

def decaptcha( filenames ):
	# Invoke your model here to make predictions on the images

	X_test = []
	for img_path in filenames:
		sample_img = cv2.imread(img_path)
		sample_img = sample_img[15:95,365:445,:]
		hsv_image = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
		_,_,v = np.mean(hsv_image[:][79], axis=0)
		lower_threshold = np.array([0, 0, 0]).astype(np.uint8)
		upper_threshold = np.array([179, 255, 0.9*v]).astype(np.uint8)
		mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
		segmented_image = cv2.bitwise_and(sample_img, sample_img, mask=mask)
		
		gray_img = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
		_, binary_image = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		final_img = cv2.erode(binary_image,kernel)

		X_test.append(final_img)
	
	X_test = np.array(X_test)
	X_test = X_test.reshape(X_test.shape[0], -1)
	X_test = X_test/255

	with open('model.pkl', 'rb') as file:
		model = pickle.load(file)

	predictions = model.predict(X_test)

	labels = ["EVEN" if val == 0 else "ODD" for val in predictions]
	
	return labels