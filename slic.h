#ifndef SLIC_H
#define SLIC_H
#include <opencv2/core.hpp>
#include <array>

namespace superpixel {
    struct PixelFeature {
        PixelFeature() {
            labValue = {};
            xy = cv::Point(0,0);
        }

        std::array<double, 3> labValue;
        cv::Point xy;
    };

    class SLIC {
     public:
      SLIC();
      void getSupperpixels(cv::Mat input, const int numOfSuperpixels, const int compactness);
     private:
      double distance(PixelFeature f1, PixelFeature f2, const int compactness, const int S);
      cv::Point getLocalMinimum(cv::Mat input, const int x, const int y, const int n);
    };
}
#endif // SLIC_H
