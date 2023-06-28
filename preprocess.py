import numpy as np
import time

from collections import OrderedDict
import dlib
import cv2


#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)


class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align(self, image, gray, rect):
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		
		#simple hack ;)
		if (len(shape)==68):
			# extract the left and right eye (x, y)-coordinates
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		else:
			(lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
			
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
			int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		# return the aligned face
		return output


class Preprocess:
    def __init__(self, GPU = True):
        self.IMG_SIZE = 224

        self.GPU = GPU
        if GPU:
            self.detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        else:
            self.detector = dlib.get_frontal_face_detector()        # for CPU

        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')     
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.IMG_SIZE, desiredFaceHeight=self.IMG_SIZE)


    def run_with_CPU(self, image):
        image = self.autoAdjustments_with_convertScaleAbs(image, .005, .995)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        results = self.detector(gray, 1)
        
        # loop over the face detections
        for rect in results:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            faceAligned = self.fa.align(image, gray, rect)

            return faceAligned
                
              
    def run_with_GPU(self, image):

        image = self.autoAdjustments_with_convertScaleAbs(image, .005, .995)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale image
        # rects = detector(gray, 2)
        start = time.time()
        results = self.detector(image, 2)
        print('detector ', time.time() - start)

        results = [r.rect for r in results]  

        # loop over the face detections
        for rect in results:
            start = time.time()
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            faceAligned = self.fa.align(image, gray, rect)

            print(time.time() - start)

            return faceAligned
                
                
                
    def run(self, image):
        if self.GPU:
            faceAligned = self.run_with_GPU(image)
        else:
            faceAligned = self.run_with_CPU(image)
        
        return faceAligned   
            
                
    def autoAdjustments_with_convertScaleAbs(self, img, low, high):
      alow =  np.quantile(img, low)
      ahigh = np.quantile(img, high)
      #print(f'low {low} max {high}')
      amax = 255
      amin = 0
      # calculate alpha, beta
      alpha = ((amax - amin) / (ahigh - alow))
      beta = amin - alow * alpha
      # perform the operation g(x,y)= α * f(x,y)+ β
      new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

      return new_img                


if __name__ == "__main__":
    preprocessor = Preprocess(GPU=True)
    preprocessor.run()

# Reference:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/    