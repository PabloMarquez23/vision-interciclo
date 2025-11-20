// src/main.cpp - VERSIÓN FINAL CON MODELO COMPATIBLE 17 CAPAS
#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"
#include "dnn_denoising.hpp" 

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp> // Para NLMeans por si acaso

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
  bool useDNN = false;
  
  try {
    // 1. CARGAR LA RED NEURONAL (DnCNN Compatible)
    try {
      // Nombre del archivo generado con PyTorch antiguo
      denoiserPtr = new DnnDenoiser("../models/dncnn_compatible.onnx"); 
      cout << "[INFO] Modelo DnCNN (17 capas) cargado correctamente.\n";
      useDNN = true;

    } catch (const cv::Exception& e) {
      cerr << "[ALERTA] El DNN falló (esto no debería pasar con el modelo compatible): " << e.what() << "\n";
      if (denoiserPtr) { delete denoiserPtr; denoiserPtr = nullptr; }
      useDNN = false;
    }
    
    // 2) Carga volumen y extrae corte ITK
    cout << "Cargando DICOM series desde: " << dicomDir << " (Slice Z=" << zIndex << ")\n";
    auto vol   = loadDicomSeries(dicomDir);
    auto slice = extractSlice(vol.image, zIndex);

    // 3) HU (float) desde ITK
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f = itk2cv32fHU(slice, &huMin, &huMax);
    if (hu32f.empty()) CV_Error(Error::StsError, "hu32f vacío");
    
    // 4) Generar imagen 8-bit base (ORIGINAL CON RUIDO)
    Mat g8_base = huTo8u(hu32f, 40.0f, 400.0f);

    // ==========================================================
    // EVIDENCIA DE REDUCCIÓN DE RUIDO
    // ==========================================================
    
    // A) CLÁSICA (Gaussiano)
    Mat g8_classical;
    GaussianBlur(g8_base, g8_classical, Size(5, 5), 0, 0); 

    // B) AVANZADA (DNN Real)
    Mat g8_advanced;
    
    if (useDNN && denoiserPtr) {
        // ¡AQUÍ DEBERÍA ENTRAR AHORA!
        cout << "[PROCESANDO] Ejecutando reducción de ruido con Deep Learning (DnCNN)...\n";
        g8_advanced = denoiserPtr->denoise(g8_base); 
    } else {
        // Fallback solo si algo salió muy mal
        cout << "[PROCESANDO] Usando NLMeans (Fallback)...\n";
        fastNlMeansDenoising(g8_base, g8_advanced, 10, 7, 21);
    }

    // Usamos la imagen limpia para el overlay
    Mat g8 = g8_advanced; 

    // 6) Segmentación en HU reales (Músculo/Grasa/Hueso - Usando suavizado interno)
    AnatomyMasks anat = generateAnatomicalMasksHU(hu32f); 
    Mat overlayAnat   = colorizeAndOverlay(g8, anat);
    
    // ... Estadísticas ...
    Mat bodyMask = (hu32f > -300.f);
    bodyMask.convertTo(bodyMask, CV_8U, 255);
    const double bodyPx = (double)countNonZero(bodyMask);
    
    // Guardado
    std::filesystem::create_directories("outputs/final");
    
    imwrite("outputs/final/slice8u_0_original_noisy.png", g8_base); 
    imwrite("outputs/final/slice8u_1_classical_gaussian.png", g8_classical); 
    
    // Esta imagen ahora será la producida por la IA Real
    imwrite("outputs/final/slice8u_2_dnn_result.png", g8_advanced); 
    
    imwrite("outputs/final/overlay_anatomical_FINAL.png", overlayAnat);
    
    // Imprimir estadísticas
    auto pct_in_body = [&](const Mat& m){
      if (bodyPx < 1.0) return 0.0;
      Mat inter; bitwise_and(m, bodyMask, inter);
      return 100.0 * (double)countNonZero(inter) / bodyPx;
    };
    cout << "\n--- ESTADÍSTICAS ---\n";
    cout << "Grasa: " << pct_in_body(anat.fat) << "% | Músculo: " << pct_in_body(anat.muscle_tendon) 
         << "% | Hueso: " << pct_in_body(anat.bones) << "%\n";

    cout << "Guardado exitoso.\n";
    imshow("Overlay Final", overlayAnat);
    waitKey(0);

  } catch (const std::exception& e) {
    cerr << "Error Fatal: " << e.what() << "\n";
    if (denoiserPtr) delete denoiserPtr;
    return 2;
  }
  
  if (denoiserPtr) delete denoiserPtr;
  return 0;
}