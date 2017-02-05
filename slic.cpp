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
        std::vector<PixelFeature> clusterCenters(numOfSuperpixels);
        int x = 0;
        int y = 0;
        const int n = 3;

        //Init clusters
        for (size_t i = 0; i < clusterCenters.size(); ++i) {
            //Perturb cluster center to the lowest gradient in an n x n neighborhood
            cv::Point bestPoint = getLocalMinimum(labSpaceInput, x*S, y*S, n);
            PixelFeature temp;
            temp.labValue[0] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(bestPoint.y, bestPoint.x)[0]);
            temp.labValue[1] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(bestPoint.y, bestPoint.x)[1]);
            temp.labValue[2] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(bestPoint.y, bestPoint.x)[2]);

            temp.xy = bestPoint;
            clusterCenters[i] = temp;
            x += 1;
            if ((x*S) >= labSpaceInput.cols) {
                x = 0;
                y += 1;
            }
        }

        double error = 0;
        const double treshold = 10;
        do{
            cv::Mat distances(labSpaceInput.size(), CV_64FC1, std::numeric_limits<double>::max());
            cv::Mat clusterIndex(labSpaceInput.size(), CV_32SC1);
            for (size_t itr = 0; itr < clusterCenters.size(); ++itr) {
                cv::Point roiCenter = clusterCenters[itr].xy;
                for (y = roiCenter.y-S; y < roiCenter.y+S; ++y) {
                    for (x = roiCenter.x-S; x < roiCenter.x+S; ++x) {
                        if ((x < 0 ) || (x >= labSpaceInput.cols) || (y < 0) || (y >= labSpaceInput.rows))
                            continue;
                        PixelFeature tempPoint;
                        tempPoint.xy = cv::Point(x, y);
                        tempPoint.labValue[0] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[0]);
                        tempPoint.labValue[1] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[1]);
                        tempPoint.labValue[2] = static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[2]);

                        const double dis = distance(clusterCenters[itr], tempPoint, compactness, S);
                        if (dis < distances.at<double>(y, x)) {
                            distances.at<double>(y, x) = dis;
                            clusterIndex.at<int>(y, x) = static_cast<int>(itr);
                        }
                    }
                }
            }

            //New cluster centers are calculated as the averag labxy PixelFeature (vector) of all pixels belonging to the cluster
            std::vector<PixelFeature> newClusterCenters(clusterCenters.size());
            std::vector<int> numOfPixels(clusterCenters.size(), 0);
            for (y = 0; y < clusterIndex.rows; ++y) {
                for (x = 0; x < clusterIndex.cols; ++x) {
                    newClusterCenters[clusterIndex.at<int>(y, x)].labValue[0] += static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[0]);
                    newClusterCenters[clusterIndex.at<int>(y, x)].labValue[1] += static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[1]);
                    newClusterCenters[clusterIndex.at<int>(y, x)].labValue[2] += static_cast<double>(labSpaceInput.at<cv::Vec3b>(y, x)[2]);
                    newClusterCenters[clusterIndex.at<int>(y, x)].xy.x += x;
                    newClusterCenters[clusterIndex.at<int>(y, x)].xy.y += y;
                    numOfPixels[clusterIndex.at<int>(y, x)] += 1;
                }
            }

            //Normalize
            for (size_t itr = 0; itr < newClusterCenters.size(); ++itr) {
                newClusterCenters[itr].labValue[0] /= numOfPixels[itr];
                newClusterCenters[itr].labValue[1] /= numOfPixels[itr];
                newClusterCenters[itr].labValue[2] /= numOfPixels[itr];
                newClusterCenters[itr].xy.x /= numOfPixels[itr];
                newClusterCenters[itr].xy.y /= numOfPixels[itr];
            }

            clusterCenters = newClusterCenters;
            //TODO Improvment calculate convergation as proposed in the original paper
            error++;
        }while(error < treshold);

        //TODO enforce conectivity
    }

    double SLIC::distance(PixelFeature f1, PixelFeature f2, const int compactness, const int S)
    {
        double dlab = sqrt(pow(f1.labValue[0]-f2.labValue[0], 2)+pow(f1.labValue[1]-f2.labValue[1], 2)+pow(f1.labValue[2]-f2.labValue[2], 2));
        double dxy = sqrt(pow(f1.xy.x-f2.xy.x, 2)+pow(f1.xy.y-f2.xy.y, 2));
        return dlab + (static_cast<double>(compactness)/static_cast<double>(S))*dxy;
    }

    cv::Point SLIC::getLocalMinimum(cv::Mat input, const int x, const int y, const int n)
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