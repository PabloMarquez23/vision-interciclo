#ifndef HIGHLIGHT_HPP
#define HIGHLIGHT_HPP

#include <opencv2/core/core.hpp> 

// Estructura para contener las 3 máscaras de tejidos
struct AnatomyMasks {
    cv::Mat fat;
    cv::Mat muscle_tendon;
    cv::Mat bones;
};

// Declaraciones de funciones (Firmas)
cv::Mat huTo8u(const cv::Mat& hu32f, float window_center, float window_width);
AnatomyMasks generateAnatomicalMasksHU(const cv::Mat& hu32f);
cv::Mat colorizeAndOverlay(const cv::Mat& slice8u, const AnatomyMasks& m);

#endif // HIGHLIGHT_HPP