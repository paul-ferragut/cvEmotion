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
		//get the ofCvColorImage as a cv::Mat image to pass to the classifier
		//auto cvMat = cv::cvarrToMat(colorImg.getCvImage());
		//frame = cvMat;

		//Run Face Detection and draw bounding box
		//face_detector.detectFace(cvMat); //P
		//face_detector.detectFace(cvMat);

		//image_ROI = face_detector.drawBoundingBoxOnFrameOF(cvMat); //
		// Draw bounding box to frame
		//image_and_ROI = face_detector.drawBoundingBoxOnFrame(cvMat);

		// Get Image ROIs
		//std::vector<cv::Mat> roi_image = image_and_ROI.getROI();
		//vector<cv::Mat> roi_image = image_ROI;
		//cout << "grabber" << endl;
		if (finder.blobs.size() > 0) {
			// Preprocess image ready for model
			//image_and_ROI.preprocessROI();
			// Make Prediction
			vector< cv::Mat> matCv;
			for (unsigned int i = 0; i < finder.blobs.size(); i++) {
				//colorImg.crop
					//ofxCvGrayscaleImage copyImg;
					//copyImg.allocate(bwImg.width, bwImg.height);
					//copyImg = bwImg;
					//colorImg.setROI(finder.blobs[i].boundingRect);
					//copyImg= colorImg.getRoiPixels();
					
					CvRect rectOF;
					rectOF.x = finder.blobs[i].boundingRect.x;
					rectOF.y = finder.blobs[i].boundingRect.y;
					rectOF.width = finder.blobs[i].boundingRect.width;
					rectOF.height = finder.blobs[i].boundingRect.height;


					//cvSetImageROI(copyImg.getCvImage(), rectOF);


					//copyImg = copyImg.getRoiPixels();


					cv::Mat t = colorImg.getCvMat();
					cv::Mat c = t(rectOF);
					cv::Mat processed_image;
					cv::Mat gray_image;
					cv::cvtColor(c, gray_image, cv::COLOR_BGR2GRAY);
					cv::resize(gray_image, processed_image, cv::Size(48, 48));
					// Convert image pixels from between 0-255 to 0-1
					processed_image.convertTo(processed_image, CV_32FC3, 1.f / 255);

					//cvCopy(colorImg, copyImg);
					//cvResetImageROI(colorImg);

					// convert to grayscale 
					//cv::Mat gray_image;
					//cv::cvtColor(_roi_image[i], gray_image, cv::COLOR_BGR2GRAY);
					// Resize the ROI to model input size
					//cv::resize(gray_image, processed_image, cv::Size(48, 48));
					// Convert image pixels from between 0-255 to 0-1
					//processed_image.convertTo(processed_image, CV_32FC3, 1.f / 255);

					
					matCv.push_back(processed_image);

					//copyImg.getRoiPixels(finder.blobs[i].boundingRect);
					//copyImg.setRoiFromPixels(finder.blobs[i].boundingRect)
					//copyImg.setFromPixels(colorImg.set;
					//finder.blobs[i].boundingRect;

			}
			vector<string>emotion_prediction = model.predict(matCv);
			// Add prediction text to the output video frame
			for (int i = 0; i < emotion_prediction.size(); i++) {
				cout << emotion_prediction[i] << endl;

			}
			//image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
		}
		//cout << "finder" << endl;


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
