// src/main.cpp
#include "itk_loader.hpp"
#include "itk_opencv_bridge.hpp"
#include "highlight.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

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

  try {
    // 1) Carga volumen y extrae corte ITK
    cout << "Cargando DICOM series desde: " << dicomDir << " (Slice Z=" << zIndex << ")\n";
    auto vol   = loadDicomSeries(dicomDir);
    auto slice = extractSlice(vol.image, zIndex);

    // 2) HU (float) desde ITK
    double huMin = 0.0, huMax = 0.0;
    Mat hu32f = itk2cv32fHU(slice, &huMin, &huMax);   // CV_32F en HU
    if (hu32f.empty()) CV_Error(Error::StsError, "hu32f vacío");
    cout << "Rango HU del slice: " << huMin << " a " << huMax << endl;

    // 3) Visual base en 8-bit (ventana músculo típica: C=40, W=400)
    Mat g8 = huTo8u(hu32f, 40.0f, 400.0f);

    // 4) Intermedios (opcional debug)
    HighlightResult base = highlightROI(g8);

    // 5) Segmentación en HU reales (Músculo/Grasa/Hueso)
    AnatomyMasks anat = generateAnatomicalMasksHU(hu32f);
    Mat overlayAnat   = colorizeAndOverlay(g8, anat);

    // 5.1) Máscara de cuerpo para estadísticas (HU > -300)
    Mat bodyMask = (hu32f > -300.f);
    bodyMask.convertTo(bodyMask, CV_8U, 255);
    const double bodyPx = (double)countNonZero(bodyMask);
    cout << "Pixeles en cuerpo (HU>-300): " << (long long)bodyPx << endl;

    // 5.2) Métricas globales y conteos de depuración
    double meanHU = cv::mean(hu32f, hu32f > -1024)[0];
    cout << "Media global HU = " << meanHU << endl;

    // 5.3) Depuración rápida de rangos HU (sobre todo el frame)
    const long long cntFatHU    = countNonZero( (hu32f >= -190.f) & (hu32f <=  -50.f) );
    const long long cntMuscleHU = countNonZero( (hu32f >=  -10.f) & (hu32f <=  120.f) );
    const long long cntBoneHU   = countNonZero( (hu32f >=  300.f) & (hu32f <= 3000.f) );
    cout << "Conteos por HU (globales):  grasa[-190,-50]=" << cntFatHU
         << "  musc[-10,120]=" << cntMuscleHU
         << "  hueso[>=300]=" << cntBoneHU << endl;

    // 6) Salidas a disco
    std::filesystem::create_directories("outputs/intermediates");
    std::filesystem::create_directories("outputs/final");

    if (!base.eq.empty())       imwrite("outputs/intermediates/equalized.png", base.eq);
    if (!base.denoised.empty()) imwrite("outputs/intermediates/denoised.png", base.denoised);
    if (!base.edges.empty())    imwrite("outputs/intermediates/edges.png", base.edges);
    if (!base.mask.empty())     imwrite("outputs/intermediates/body_mask_gray.png", base.mask);

    // Máscaras finales (sobre el cuerpo)
    if (!anat.fat.empty())            imwrite("outputs/intermediates/mask_fat_final.png", anat.fat);
    if (!anat.muscle_tendon.empty())  imwrite("outputs/intermediates/mask_muscle_tendon_final.png", anat.muscle_tendon);
    if (!anat.bones.empty())          imwrite("outputs/intermediates/mask_bones_final.png", anat.bones);

    if (!base.finalVis.empty()) imwrite("outputs/final/overlay_baseline.png", base.finalVis);
    imwrite("outputs/final/overlay_anatomical_FINAL.png", overlayAnat);
    imwrite("outputs/final/slice8u.png", g8);
    imwrite("outputs/intermediates/body_mask_hu_gt-300.png", bodyMask);

    // 7) Estadísticos: porcentajes SOBRE EL CUERPO (lo correcto para el análisis)
    auto pct_in_body = [&](const Mat& m){
      if (bodyPx < 1.0) return 0.0;
      Mat inter; bitwise_and(m, bodyMask, inter);
      return 100.0 * (double)countNonZero(inter) / bodyPx;
    };
    cout << "\n--- ESTADÍSTICAS SOBRE EL CUERPO ---\n";
    cout << "Áreas sobre cuerpo (%)  Grasa: " << pct_in_body(anat.fat)
         << "  | Músculo/Tendón: " << pct_in_body(anat.muscle_tendon)
         << "  | Hueso: "  << pct_in_body(anat.bones) << endl;
    cout << "------------------------------------\n";

    cout << "Guardado en outputs/ ...\n";

    imshow("Overlay anatomico (3 zonas)", overlayAnat);
    waitKey(0);

  } catch (const std::exception& e) {
    cerr << "Error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}