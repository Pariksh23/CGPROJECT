#include <iostream>
using namespace std;
#include "opencv2/opencv.hpp"
using namespace cv;

int main() {
    // Load the pre-trained face detection classifier
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face cascade!\n";
        return -1;
    }

    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening webcam!\n";
        return -1;
    }

    // Create a window to display the detected faces
    namedWindow("Face Detection", WINDOW_NORMAL);

    // Main loop
    while (true) {
        // Read frame from webcam
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error reading frame from webcam!\n";
            break;
        }

        // Convert frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces in the frame
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Draw rectangles around the detected faces
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Display the frame with detected faces
        imshow("Face Detection", frame);

        // Check for exit key press
        if (waitKey(1) == 27) {
            break; // Exit loop if ESC is pressed
        }
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
