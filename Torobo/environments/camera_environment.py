import cv2

class Camera:
	def camera_establish(self):
		self.cap = cv2.VideoCapture(0)
		return self.cap

	def camera_update(self, cap):
		frame = cap.read()[1]
		frame = cv2.resize(frame, (240, 240))
		return frame

	def display(self):
		while True:
			frame = self.camera_update(self.cap)
			cv2.imshow('frame', frame)
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
		cv2.destroyAllWindows()

if __name__ == '__main__':
	cam = Camera()
	cam.camera_establish()
	cam.display()