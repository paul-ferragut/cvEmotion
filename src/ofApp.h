#pragma once

#include "FaceDetector.h"
#include "Image.h"
#include "Model.h"

#include "ofxOpenCv.h"
#include "ofMain.h"


const std::string TENSORFLOW_MODEL_PATH = "C:/Users/plesk/Documents/Coding/of_v20230328_vs_release/apps/myApps/cvEmotion/bin/data/model/tensorflow_model.pb";
const std::string FACE_DETECTOR_MODEL_PATH = "C:/Users/plesk/Documents/Coding/of_v20230328_vs_release/apps/myApps/cvEmotion/bin/data/model/haarcascade_frontalface_alt2.xml";


class ofApp : public ofBaseApp{

	public:
		
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		ofVideoGrabber grabber;
		ofxCvColorImage	colorImg;
		
		cv::Mat frame;

		Model model{TENSORFLOW_MODEL_PATH};// error variable "TENSORFLOW_MODEL_PATH" is not a type name

		FaceDetector face_detector;
		Image image_and_ROI;

};
