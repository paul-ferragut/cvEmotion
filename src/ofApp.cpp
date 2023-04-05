#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	finder.setup("model/haarcascade_frontalface_alt2.xml");
	//finder.findHaarObjects(img);

	model.modelCuda(true);
	grabber.setDeviceID(1);
	grabber.initGrabber(640, 480);
	colorImg.allocate(640, 480);
}

//--------------------------------------------------------------
void ofApp::update(){


	grabber.update();
	if (grabber.isFrameNew()) {

		//get the ofPixels and convert to an ofxCvColorImage
		auto pixels = grabber.getPixels();
		colorImg.setFromPixels(pixels);
		ofxCvGrayscaleImage bwImg;
		bwImg = colorImg;
		finder.findHaarObjects(bwImg);

		if (finder.blobs.size() > 0) {
			// Make Prediction
			vector< cv::Mat> matCv;
			for (unsigned int i = 0; i < finder.blobs.size(); i++) {

					CvRect rectOF;
					rectOF.x = finder.blobs[i].boundingRect.x;
					rectOF.y = finder.blobs[i].boundingRect.y;
					rectOF.width = finder.blobs[i].boundingRect.width;
					rectOF.height = finder.blobs[i].boundingRect.height;

					cv::Mat t = colorImg.getCvMat();
					cv::Mat c = t(rectOF);
					cv::Mat processed_image;
					cv::Mat gray_image;
					cv::cvtColor(c, gray_image, cv::COLOR_BGR2GRAY);
					cv::resize(gray_image, processed_image, cv::Size(48, 48));
					// Convert image pixels from between 0-255 to 0-1
					processed_image.convertTo(processed_image, CV_32FC3, 1.f / 255);				
					matCv.push_back(processed_image);

			}
			emotion_prediction= model.predict(matCv);
			//vector<string>emotion_prediction = model.predict(matCv);
			// Add prediction text to the output video frame
			//for (int i = 0; i < emotion_prediction.size(); i++) {
			//	cout << emotion_prediction[i] << endl;
			//}
		}
	}

	
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofFill();
	grabber.draw(0,0);
	ofNoFill();
	for (unsigned int i = 0; i < finder.blobs.size(); i++) {
		ofRectangle cur = finder.blobs[i].boundingRect;
		ofDrawRectangle(cur.x, cur.y, cur.width, cur.height);
		
	}

	for (int i = 0; i < emotion_prediction.size(); i++) {
		ofDrawBitmapString(emotion_prediction[i], finder.blobs[i].boundingRect.x, finder.blobs[i].boundingRect.y);
	}
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
