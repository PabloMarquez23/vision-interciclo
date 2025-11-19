#include "highlight.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;

// ==========================================================
// IMPLEMENTACIÓN DE generateAnatomicalMasksHU (LA FUNCIÓN FALTANTE EN EL LINKER)
// ==========================================================
AnatomyMasks generateAnatomicalMasksHU(const Mat& hu32f) {
    AnatomyMasks m;

    // Rangos de HU
    m.fat = (hu32f >= -190.f) & (hu32f <= -30.f);
    m.muscle_tendon = (hu32f >= 10.f) & (hu32f <= 120.f);
    m.bones = hu32f >= 200.f;

    // Convertir las máscaras booleanas a CV_8U (0, 255)
    m.fat.convertTo(m.fat, CV_8U, 255);
    m.muscle_tendon.convertTo(m.muscle_tendon, CV_8U, 255);
    m.bones.convertTo(m.bones, CV_8U, 255);
    
    return m;
}


// ==========================================================
// IMPLEMENTACIÓN DE colorizeAndOverlay (Superposición transparente con colores sólidos)
// ==========================================================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
    Mat base_bgr;
    if (slice8u.channels() == 1) cvtColor(slice8u, base_bgr, COLOR_GRAY2BGR); 
    else base_bgr = slice8u.clone();

    // Definición de Colores BGR (Hueso celeste)
    const Scalar COLOR_FAT = Scalar(255, 128,   0); 
    const Scalar COLOR_MUSCLE = Scalar(  0,  69, 255); 
    const Scalar COLOR_BONE = Scalar(255, 100,   0); 

    Mat colored_fat = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_fat.setTo(COLOR_FAT, m.fat);

    Mat colored_muscle = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_muscle.setTo(COLOR_MUSCLE, m.muscle_tendon);

    Mat colored_bone = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_bone.setTo(COLOR_BONE, m.bones);

    Mat final_overlay = base_bgr.clone();

    // Transparencia (puedes ajustar alpha, 0.6 = 60% color)
    double alpha = 0.6; 
    double beta = 1.0 - alpha; 

    // Superponemos las máscaras de forma jerárquica
    addWeighted(final_overlay, beta, colored_fat, alpha, 0, final_overlay, final_overlay.depth());
    addWeighted(final_overlay, beta, colored_muscle, alpha, 0, final_overlay, final_overlay.depth());
    addWeighted(final_overlay, beta, colored_bone, alpha, 0, final_overlay, final_overlay.depth());

    return final_overlay;
}