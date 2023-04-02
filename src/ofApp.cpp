#include "ofApp.h"
#include "Model.h"



//Model model(TENSORFLOW_MODEL_PATH);// No symbols loaded errors
//--------------------------------------------------------------
void ofApp::setup(){
	

	model.modelCuda(true);
	grabber.initGrabber(640, 480);	
	
	//

	
}

//--------------------------------------------------------------
void ofApp::update(){

	ofSetFrameRate(240);
	//Model model(TENSORFLOW_MODEL_PATH); //works but local scope
	ofBackground(30, 30, 30);

	grabber.update();
	if (grabber.isFrameNew()) {

		//get the ofPixels and convert to an ofxCvColorImage
		auto pixels = grabber.getPixels();
		colorImg.setFromPixels(pixels);

		//get the ofCvColorImage as a cv::Mat image to pass to the classifier
		auto cvMat = cv::cvarrToMat(colorImg.getCvImage());
		frame = cvMat;

		//Run Face Detection and draw bounding box
		face_detector.detectFace(cvMat);
		//face_detector.detectFace(cvMat);

		// Draw bounding box to frame
		image_and_ROI = face_detector.drawBoundingBoxOnFrame(cvMat);

		// Get Image ROIs
		std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

		
		if (roi_image.size() > 0) {
			// Preprocess image ready for model
			image_and_ROI.preprocessROI();
			// Make Prediction
			std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
			// Add prediction text to the output video frame

			for (int i = 0; i < emotion_prediction.size(); i++) {
				cout<< emotion_prediction[i] << endl;
			
			}
			image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
		}
		

	}
	ofSetWindowTitle("framerate"+ofToString(ofGetFrameRate()));
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofSetColor(255);
	grabber.draw(0, 0);

	//image_and_ROI.getFrame();
	//draw the detected objects on top of the webcam image 
	ofNoFill();
	ofSetColor(255, 0, 255);

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}



//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
