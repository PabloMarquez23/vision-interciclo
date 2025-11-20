#include "highlight.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

// ==========================================================
// IMPLEMENTACIÓN PURA (Sin suavizado interno)
// ==========================================================
AnatomyMasks generateAnatomicalMasksHU(const Mat& huInput) {
    AnatomyMasks m;

    // Usamos directamente la matriz de entrada (sea cruda o suavizada)
    // Rangos de HU estándar
    Mat raw_fat    = (huInput >= -190.f) & (huInput <= -30.f);
    Mat raw_muscle = (huInput >= 10.f)   & (huInput <= 120.f);
    Mat raw_bone   = (huInput >= 200.f);

    // Convertir a 8-bit
    raw_fat.convertTo(m.fat, CV_8U, 255);
    raw_muscle.convertTo(m.muscle_tendon, CV_8U, 255);
    raw_bone.convertTo(m.bones, CV_8U, 255);

    // Mantenemos la limpieza morfológica (para unir regiones), 
    // pero si la entrada es muy ruidosa, esto no será suficiente para arreglarla.
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernelLg = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    morphologyEx(m.bones, m.bones, MORPH_CLOSE, kernel);

    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_OPEN, kernel);
    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_CLOSE, kernelLg);

    morphologyEx(m.fat, m.fat, MORPH_OPEN, kernel);

    // Jerarquía
    m.muscle_tendon.setTo(0, m.bones);
    m.fat.setTo(0, m.muscle_tendon);
    m.fat.setTo(0, m.bones);

    return m;
}

// ==========================================================
// IMPLEMENTACIÓN DE colorizeAndOverlay (Se mantiene igual)
// ==========================================================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
    Mat base_bgr;
    if (slice8u.channels() == 1) cvtColor(slice8u, base_bgr, COLOR_GRAY2BGR); 
    else base_bgr = slice8u.clone();

    // Hueso: CYAN BRILLANTE
    const Scalar COLOR_BONE = Scalar(255, 255, 0); 
    // Músculo: MAGENTA / ROSA
    const Scalar COLOR_MUSCLE = Scalar(128, 0, 255); 
    // Grasa: AMARILLO LIMA
    const Scalar COLOR_FAT = Scalar(0, 255, 255); 

    Mat colored_fat = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_fat.setTo(COLOR_FAT, m.fat);

    Mat colored_muscle = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_muscle.setTo(COLOR_MUSCLE, m.muscle_tendon);

    Mat colored_bone = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_bone.setTo(COLOR_BONE, m.bones);

    Mat final_overlay = base_bgr.clone();

    // Transparencia
    double alpha = 0.60; 
    
    if (countNonZero(m.fat) > 0)
        addWeighted(final_overlay, 1.0, colored_fat, alpha, 0, final_overlay);
        
    if (countNonZero(m.muscle_tendon) > 0)
        addWeighted(final_overlay, 1.0, colored_muscle, alpha, 0, final_overlay);
        
    if (countNonZero(m.bones) > 0)
        addWeighted(final_overlay, 1.0, colored_bone, alpha, 0, final_overlay);

    return final_overlay;
}