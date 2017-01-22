#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
    cv::Mat testMatrix(512, 512, CV_8UC3, cv::Scalar(0));
    cv::imshow("test", testMatrix);
    cv::waitKey();
    return 0;
}
