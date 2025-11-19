// src/highlight.hpp
#pragma once
#include <opencv2/core.hpp>

using namespace cv;

// ===== Resultados intermedios (debug/visual) =====
struct HighlightResult {
  cv::Mat eq;
  cv::Mat denoised;
  cv::Mat edges;
  cv::Mat mask;
  cv::Mat overlay;
  cv::Mat finalVis;
};

// ===== Máscaras anatómicas =====
struct AnatomyMasks {
  cv::Mat fat;            // Grasa
  cv::Mat muscle_tendon;  // Músculo / tendón
  cv::Mat bones;          // Hueso
};

// Intermedios (gris 8-bit)
HighlightResult highlightROI(const cv::Mat& gray);

// Segmentación principal en HU reales (CV_32F en unidades HU)
AnatomyMasks generateAnatomicalMasksHU(const cv::Mat& hu32f);

// Overlay coloreado sobre una imagen 8-bit (BGR o GRAY)
cv::Mat colorizeAndOverlay(const cv::Mat& slice8u, const AnatomyMasks& m);