#include "ofMain.h"
#include "ofApp.h"
#include "Model.h"

//========================================================================
int main( ){


	//Model model(TENSORFLOW_MODEL_PATH);
	//FaceDetector face_detector;
	//Use ofGLFWWindowSettings for more options like multi-monitor fullscreen
	ofGLWindowSettings settings;
	settings.setSize(1024, 768);
	settings.windowMode = OF_WINDOW; //can also be OF_FULLSCREEN

	auto window = ofCreateWindow(settings);

	ofRunApp(window, make_shared<ofApp>());
	ofRunMainLoop();

}
