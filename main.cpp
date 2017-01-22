#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>

void usage()
{
    std::cout << "Usage: SLIC_Superpixel -i inputFile -o outputFile" << std::endl;
}

int main(int argc, char *argv[])
{
    extern char *optarg;
    extern int optopt;
    std::string inputFile, outputFile;
    bool errorOpt = false;
    int c;
    while ((c = getopt(argc, argv, ":i:o:")) != -1) {
        switch (c) {
        case 'i':
            inputFile = optarg;
            break;
        case 'o':
            outputFile = optarg;
            break;
        case ':':
            std::cout << "Option -" << (char)optopt << " requires filename." << std::endl;
            errorOpt = true;
            break;
        case '?':
            std::cout << "Unrecognized option: -" << (char)optopt << std::endl;
            errorOpt = true;
            break;
        }
    }

    if (errorOpt || argc < 5) {
        usage();
        return -1;
    }

    cv::Mat testMatrix(512, 512, CV_8UC3, cv::Scalar(0));
    cv::imshow("test", testMatrix);
    cv::waitKey();
    return 0;
}
