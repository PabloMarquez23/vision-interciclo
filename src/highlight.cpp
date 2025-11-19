#include "highlight.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

// ==========================================================
// IMPLEMENTACIÓN DE generateAnatomicalMasksHU (CON LIMPIEZA DE RUIDO)
// ==========================================================
AnatomyMasks generateAnatomicalMasksHU(const Mat& hu32f) {
    AnatomyMasks m;

    // 1. UMBRALIZACIÓN (Thresholding)
    // Usamos rangos estándar para tomografía
    Mat raw_fat    = (hu32f >= -190.f) & (hu32f <= -30.f);
    Mat raw_muscle = (hu32f >= 10.f)  & (hu32f <= 120.f);
    Mat raw_bone   = (hu32f >= 200.f);

    // Convertir a formato de imagen (0-255)
    raw_fat.convertTo(m.fat, CV_8U, 255);
    raw_muscle.convertTo(m.muscle_tendon, CV_8U, 255);
    raw_bone.convertTo(m.bones, CV_8U, 255);

    // 2. LIMPIEZA MORFOLÓGICA (CRÍTICO PARA MEJORAR VISUALIZACIÓN)
    // Esto elimina los "puntitos" sueltos y rellena huecos pequeños
    
    // Elemento estructurante (forma del pincel de limpieza)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernelLg = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // Hueso: Cierre para rellenar poros pequeños
    morphologyEx(m.bones, m.bones, MORPH_CLOSE, kernel);

    // Músculo: Apertura para quitar ruido blanco (puntos aislados) y Cierre para unir fibras
    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_OPEN, kernel);
    morphologyEx(m.muscle_tendon, m.muscle_tendon, MORPH_CLOSE, kernelLg);

    // Grasa: Apertura para quitar ruido
    morphologyEx(m.fat, m.fat, MORPH_OPEN, kernel);

    // 3. EXCLUSIÓN JERÁRQUICA (Limpieza lógica)
    // El hueso tiene prioridad sobre el músculo
    m.muscle_tendon.setTo(0, m.bones);
    
    // El músculo tiene prioridad sobre la grasa
    m.fat.setTo(0, m.muscle_tendon);
    m.fat.setTo(0, m.bones);

    return m;
}


// ==========================================================
// IMPLEMENTACIÓN DE colorizeAndOverlay (COLORES VIVOS TIPO NEÓN)
// ==========================================================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
    Mat base_bgr;
    if (slice8u.channels() == 1) cvtColor(slice8u, base_bgr, COLOR_GRAY2BGR); 
    else base_bgr = slice8u.clone();

    // --- PALETA DE COLORES DE ALTO CONTRASTE ---
    // Usamos colores brillantes para que resalten sobre el gris
    
    // Hueso: CYAN BRILLANTE (Azul eléctrico)
    const Scalar COLOR_BONE = Scalar(255, 255, 0); // BGR: Azul+Verde
    
    // Músculo: MAGENTA / ROSA FUERTE (Contrasta bien con el cyan)
    const Scalar COLOR_MUSCLE = Scalar(128, 0, 255); // BGR: Azul+Rojo
    
    // Grasa: AMARILLO LIMA
    const Scalar COLOR_FAT = Scalar(0, 255, 255); // BGR: Verde+Rojo

    // Crear capas de color
    Mat colored_fat = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_fat.setTo(COLOR_FAT, m.fat);

    Mat colored_muscle = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_muscle.setTo(COLOR_MUSCLE, m.muscle_tendon);

    Mat colored_bone = Mat::zeros(base_bgr.size(), base_bgr.type());
    colored_bone.setTo(COLOR_BONE, m.bones);

    Mat final_overlay = base_bgr.clone();

    // --- MEZCLA INTELIGENTE ---
    // Aumentamos la opacidad (alpha) para que los colores se vean más sólidos
    double alpha = 0.65; // 65% Color
    double beta = 1.0;   // 100% Imagen de fondo (para no oscurecerla)

    // Usamos addWeighted para mezclar preservando el brillo
    if (countNonZero(m.fat) > 0)
        addWeighted(final_overlay, 1.0, colored_fat, alpha, 0, final_overlay);
        
    if (countNonZero(m.muscle_tendon) > 0)
        addWeighted(final_overlay, 1.0, colored_muscle, alpha, 0, final_overlay);
        
    if (countNonZero(m.bones) > 0)
        addWeighted(final_overlay, 1.0, colored_bone, alpha, 0, final_overlay);

    return final_overlay;
}