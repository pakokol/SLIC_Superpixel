#include "slic.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

namespace superpixel {
    SLIC::SLIC()
    {

    }

    void SLIC::getSupperpixels(cv::Mat input, const int numOfSuperpixels, const int compactness)
    {
        cv::Mat labSpaceInput;
        cv::cvtColor(input, labSpaceInput, CV_BGR2Lab);
        const int S = static_cast<int>(floor(sqrt((labSpaceInput.rows * labSpaceInput.cols)/numOfSuperpixels)));
        std::vector<PixelFeature> C(numOfSuperpixels);
        int x = 0;
        int y = 0;
        const int n = 3;

        //Init clusters
        for (size_t i = 0; i < C.size(); ++i) {
            //Perturb cluster center to the lowest gradient in an n x n neighborhood
            cv::Point bestPoint = getPixleWithLowesGradient(labSpaceInput, x*S, y*S, n);
            PixelFeature temp;
            temp.labValue = labSpaceInput.at<cv::Vec3b>(bestPoint.y, bestPoint.x);
            temp.xy = bestPoint;
            C[i] = temp;
            x += 1;
            if ((x*S)>=labSpaceInput.cols) {
                x = 0;
                y += 1;
            }
        }

        //TODO cluster the pixels

    }

    double SLIC::distance(PixelFeature f1, PixelFeature f2, const int compactness, const int S)
    {
        double dlab = sqrt(pow(f1.labValue[0]-f2.labValue[0], 2)+pow(f1.labValue[1]-f2.labValue[1], 2)+pow(f1.labValue[2]-f2.labValue[2], 2));
        double dxy = sqrt(pow(f1.xy.x-f2.xy.x, 2)+pow(f1.xy.y-f2.xy.y, 2));
        return dlab + (static_cast<double>(compactness)/static_cast<double>(S))*dxy;
    }

    cv::Point SLIC::getPixleWithLowesGradient(cv::Mat input, const int x, const int y, const int n)
    {
        auto lambdaGrad = [](cv::Mat input, int x, int y){
                                cv::Vec3b xDiff = input.at<cv::Vec3b>(y, x + 1) - input.at<cv::Vec3b>(y, x - 1);
                                cv::Vec3b yDiff = input.at<cv::Vec3b>(y + 1, x) - input.at<cv::Vec3b>(y - 1, x);
                                int xSum = 0;
                                int ySum = 0;

                                for (size_t i = 0; i < xDiff.cols; ++i) {
                                    xSum += pow(xDiff[i], 2);
                                    ySum += pow(yDiff[i], 2);
                                }

                                return xSum + ySum;
                        };

        double minGradient = std::numeric_limits<double>::max();
        int min_x = x;
        int min_y = y;
        for (int itrY = y - static_cast<int>(static_cast<double>(n)/2.0); itrY < y + static_cast<int>(static_cast<double>(n)/2.0); ++itrY){
            for (int itrX = x - static_cast<int>(static_cast<double>(n)/2.0); itrX < x + static_cast<int>(static_cast<double>(n)/2.0); ++itrX){
                if (itrY < 1 || itrX < 1)
                    continue;

                const double gradient = lambdaGrad(input, itrX, itrY);
                if (gradient < minGradient) {
                    minGradient = gradient;
                    min_x = itrX;
                    min_y = itrY;
                }
            }
        }

        return cv::Point(min_x, min_y);
    }
}
