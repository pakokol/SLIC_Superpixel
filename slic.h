#ifndef SLIC_H
#define SLIC_H
#include <opencv2/core.hpp>
#include <array>

namespace superpixel {
    struct PixelFeature {
        //Constructor
        PixelFeature(): labValue{}, xy(0,0) { }
        //Public variables
        std::array<double, 3> labValue;
        cv::Point xy;
    };

    class SLIC {
     public:
      void getSupperpixels(cv::Mat input, const int numOfSuperpixels, const int compactness, const double treshold);
     private:
      double distance(PixelFeature f1, PixelFeature f2, const int compactness, const int S);
      void enforceConnectivity(cv::Mat labels, cv::Mat &newLabels, const int numOfSuperpixels);
      cv::Point getLocalMinimum(cv::Mat input, const int x, const int y, const int n);
    };
}
#endif // SLIC_H
