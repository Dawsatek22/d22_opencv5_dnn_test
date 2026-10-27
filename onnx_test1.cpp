#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

string model = "/home/d22/cv_projects/dnn/yolo26n.onnx"; // full path
const float nms_threshold = 0.45;
const float score_threshold = 1.0;
const float confidence_thresh = 0.5;
VideoCapture cam(0);
int main() {
vector<string> classes; 

    // Open file with labels.
    string labels_filename = "/home/d22/cv_projects/dnn/data/txt/object_detection_classes_coco.txt"; // full path
    string file = labels_filename;
    ifstream ifs(file);
    if (!ifs.is_open()){
        cout<<"File " << file << " not found";
        exit(1);
    }
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

EngineType engine = ENGINE_AUTO;
Net net = readNetFromONNX(model,engine);


int input_width = 640;
int input_height = 640;

 while (true) {
Mat frame ,blob;
cam.read(frame);
if (!cam.isOpened()) {

    break;
}
    
        blobFromImage(frame, blob, 1.0 / 255.0, Size(input_width, input_height), Scalar(), false, false,CV_32F);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
int rows = outputs[0].size[1];
int dimensions = outputs[0].size[2];

outputs[0] = outputs[0].reshape(1,dimensions);

 transpose(outputs[0],outputs[0]);
 
 float *data = (float *)outputs[0].data;

 vector<int> class_ids;
 vector<float> confidences;
 
vector<Rect> boxes;

float *classes_scores = data+5;

Mat scores(1,60,CV_32FC1,classes_scores);
Point class_id;
double maxClassScore = 0.0;
minMaxLoc(scores,0,&maxClassScore, 0 , &class_id);
if (maxClassScore > score_threshold)
{
    confidences.push_back(maxClassScore);
    class_ids.push_back(class_id.x);

float x = data[0]; // center x
float y = data[1]; // center y

float w = data[2]; // center w
float h = data[3]; // center h
int left = x * frame.cols;
int top = y * frame.rows;


int width = w * frame.cols;
int height = h * frame.rows;

boxes.push_back(Rect(left,top,width,height));


data += dimensions;
    }
vector<int> nms_result;
NMSBoxes(boxes,confidences,score_threshold,nms_threshold,nms_result);
for (unsigned int i =0; i < nms_result.size(); i++) {
int idx = nms_result[i];
Mat re_frame;
resize(frame,re_frame,Size(840,640));
rectangle(re_frame,boxes[idx],Scalar(0,255,200),1);
putText(re_frame, classes[class_ids[idx]],Point(boxes[idx].x,boxes[idx].y),
1,3,Scalar(0,255,0),1);
cout << "the value"<<classes[class_ids[idx]]<< '\n';

resize(frame,re_frame,Size(840,640));
imshow("dnn_test",re_frame);
} 
 
if(waitKey(1) == 'q') {
break;

}
}
}
