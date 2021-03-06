#include <iostream>
#include <unistd.h>
#include "slic.h"
#include <opencv2/imgcodecs.hpp>

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

    cv::Mat inputMatrix = cv::imread(inputFile);
    const int numberOfSupperpixels = 400;
    const int compactness = 20;
    const double treshold = 30.0;
    superpixel::SLIC slic;
    slic.getSupperpixels(inputMatrix, numberOfSupperpixels, compactness, treshold);
    return 0;
}
