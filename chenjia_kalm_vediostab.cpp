#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#define camera_width 320 
#define camera_height 240
using namespace std;
using namespace cv;


const int HORIZONTAL_BORDER_CROP = 20;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da;
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};
//
int main(int argc, char **argv)
{

	VideoCapture cap(0);
	assert(cap.isOpened());
	cap.set(CAP_PROP_FRAME_WIDTH,camera_width);
	cap.set(CAP_PROP_FRAME_HEIGHT,camera_height);
	Mat cur, cur_grey;
	Mat prev, prev_grey;

	cap >> prev;
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
	

	double a = 0;
	double x = 0;
	double y = 0;

	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance 

	Mat T(2,3,CV_64F);

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct

	//
	int k=1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	Mat prev_grey_,cur_grey_;
	 
	while(true) {

		cap >> cur;
		if(cur.data == NULL) {
			break;
		}

		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <uchar> status;
		vector <float> err;

		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

		// weed out bad matches
		for(size_t i=0; i < status.size(); i++) {
			if(status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}

		// translation + rotation only
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

		x += dx;
		y += dy;
		a += da;

		z = Trajectory(x,y,a);
		if(k==1){
			// intial guesses
			X = Trajectory(0,0,0); //Initial estimate,  set 0
			P =Trajectory(1,1,1); //set error variance,set 1
		}
		else
		{
			//time update£¨prediction£©
			X_ = X; //X_(k) = X(k-1);
			P_ = P+Q; //P_(k) = P(k-1)+Q;
			// measurement update£¨correction£©
			K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
			P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
		}

		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;


		T.at<double>(0,0) = cos(da);
		T.at<double>(0,1) = -sin(da);
		T.at<double>(1,0) = sin(da);
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;

		Mat cur2;
		
		warpAffine(prev, cur2, T, cur.size());

		cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

		// Resize cur2 back to cur size, for better side by side comparison
		resize(cur2, cur2, cur.size());

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

		prev.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		if(canvas.cols > 1920) {
			resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
		}


		imshow("before and after", canvas);

		if((char)waitKey(10)=='e')break;
		//
		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
		k++;


	}
	return 0;
}
