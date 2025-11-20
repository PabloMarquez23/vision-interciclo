#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp" 

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp> 

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <carpeta_dicom> <z_index>\n";
    return 1;
  }

  const std::string dicomDir = argv[1];
  const unsigned int zIndex  = static_cast<unsigned int>(std::stoul(argv[2]));

  DnnDenoiser* denoiserPtr = nullptr;
  
  try {
    // ---------------------------------------------------------
    // 0. CARGAR DNN (Solo intentamos cargar, no usamos aún)
    // ---------------------------------------------------------
    try {
        denoiserPtr = new DnnDenoiser("../models/dncnn_compatible.onnx"); 
        cout << "[INIT] Modelo DnCNN cargado correctamente.\n";
    } catch (const cv::Exception& e) {
        cerr << "[ALERTA] Fallo carga DNN. Se usará NLMeans como método avanzado.\n";
    }
    
    // ---------------------------------------------------------
    // 1. PREPARAR DATOS (HU CRUDOS)
    // ---------------------------------------------------------
    cout << "[DATA] Cargando DICOM...\n";
    auto vol   = loadDicomSeries(dicomDir);
    auto slice = extractSlice(vol.image, zIndex);
    
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f_raw = itk2cv32fHU(slice, &huMin, &huMax); 
    if (hu32f_raw.empty()) CV_Error(Error::StsError, "hu32f vacío");


    // =========================================================
    // FLUJO 1: ORIGINAL (CONTROL) - SIN PROCESAMIENTO
    // =========================================================
    cout << "--- Procesando Flujo 1: Original ---\n";
    // A. Imagen Visual (Sin reducción de ruido)
    Mat img_visual_1 = huTo8u(hu32f_raw, 40.0f, 400.0f);
    
    // B. Segmentación (Sobre datos crudos)
    // Esto debería verse "sucio" si highlight.cpp no tiene suavizado interno
    AnatomyMasks masks_1 = generateAnatomicalMasksHU(hu32f_raw);
    Mat overlay_1 = colorizeAndOverlay(img_visual_1, masks_1);


    // =========================================================
    // FLUJO 2: MÉTODO CLÁSICO (GAUSSIANO)
    // =========================================================
    cout << "--- Procesando Flujo 2: Clásico (Gauss) ---\n";
    
    // A. Imagen Visual (Suavizada con Gaussiano)
    Mat img_visual_2;
    GaussianBlur(img_visual_1, img_visual_2, Size(5, 5), 1.0);
    
    // B. Segmentación (Sobre datos HU suavizados con Gaussiano)
    Mat hu32f_gauss;
    GaussianBlur(hu32f_raw, hu32f_gauss, Size(5, 5), 1.0); // Suavizamos los datos médicos
    AnatomyMasks masks_2 = generateAnatomicalMasksHU(hu32f_gauss);
    Mat overlay_2 = colorizeAndOverlay(img_visual_2, masks_2);


    // =========================================================
    // FLUJO 3: MÉTODO AVANZADO (DNN o NLMeans)
    // =========================================================
    cout << "--- Procesando Flujo 3: Avanzado (DNN) ---\n";
    Mat img_visual_3;
    Mat hu32f_dnn;

    if (denoiserPtr) {
        // SI EL DNN FUNCIONA:
        cout << "[DNN] Usando Red Neuronal para limpiar...\n";
        
        // A. Imagen Visual
        img_visual_3 = denoiserPtr->denoise(img_visual_1);

        // B. Datos Médicos (HU)
        // Truco: Normalizamos HU a 0-1, pasamos por la red, y des-normalizamos
        Mat hu_norm;
        normalize(hu32f_raw, hu_norm, 0, 255, NORM_MINMAX, CV_8U);
        Mat hu_clean_8u = denoiserPtr->denoise(hu_norm);
        hu_clean_8u.convertTo(hu32f_dnn, CV_32F);
        // Recuperar escala aproximada HU
        normalize(hu32f_dnn, hu32f_dnn, huMin, huMax, NORM_MINMAX);

    } else {
        // SI EL DNN FALLA (Fallback a NLMeans):
        cout << "[FALLBACK] Usando NLMeans (Avanzado)...\n";
        
        // A. Imagen Visual
        fastNlMeansDenoising(img_visual_1, img_visual_3, 10, 7, 21);
        
        // B. Datos Médicos (HU) - Aplicamos Gaussiano suave como proxy avanzado para segmentar
        // (NLMeans es lento en floats, usamos una aproximación de alta calidad)
        GaussianBlur(hu32f_raw, hu32f_dnn, Size(3, 3), 0.8); 
    }

    // B. Segmentación Avanzada
    AnatomyMasks masks_3 = generateAnatomicalMasksHU(hu32f_dnn);
    Mat overlay_3 = colorizeAndOverlay(img_visual_3, masks_3);


    // =========================================================
    // GUARDADO DE LAS 6 IMÁGENES
    // =========================================================
    std::filesystem::create_directories("outputs/final");
    
    // GRUPO A: IMÁGENES "LIMPIAS" (Sin segmentación)
    imwrite("outputs/final/1_Visual_Original.png", img_visual_1);
    imwrite("outputs/final/2_Visual_Clasico.png", img_visual_2);
    imwrite("outputs/final/3_Visual_Avanzado.png", img_visual_3);

    // GRUPO B: IMÁGENES SEGMENTADAS (Con colores)
    imwrite("outputs/final/4_Overlay_Original.png", overlay_1);
    imwrite("outputs/final/5_Overlay_Clasico.png", overlay_2);
    imwrite("outputs/final/6_Overlay_Avanzado.png", overlay_3);

    cout << "\n========================================================\n";
    cout << "¡GENERACIÓN EXITOSA! Revisa la carpeta outputs/final/\n";
    cout << "--------------------------------------------------------\n";
    cout << "1. Visual Original  |  4. Segmentacion Original (Mala)\n";
    cout << "2. Visual Clasico   |  5. Segmentacion Clasica\n";
    cout << "3. Visual Avanzado  |  6. Segmentacion Avanzada (DNN)\n";
    cout << "========================================================\n";

    // MOSTRAMOS LAS 6 VENTANAS
    // (Las acomodamos para que no se solapen todas)
    imshow("1. Visual Original", img_visual_1);
    imshow("2. Visual Clasico", img_visual_2);
    imshow("3. Visual Avanzado", img_visual_3);
    
    imshow("4. Overlay Original", overlay_1);
    imshow("5. Overlay Clasico", overlay_2);
    imshow("6. Overlay Avanzado", overlay_3);

    waitKey(0);

  } catch (const std::exception& e) {
    cerr << "Error Fatal: " << e.what() << "\n";
    if (denoiserPtr) delete denoiserPtr;
    return 2;
  }
  
  if (denoiserPtr) delete denoiserPtr;
  return 0;
}