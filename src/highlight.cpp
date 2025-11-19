// src/highlight.cpp - CÓDIGO FINAL DEFINITIVO CORREGIDO
#include "highlight.hpp"
#include "processing.hpp" // Asume que processing.hpp define morphCloseK, etc.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// ===================== HELPERS: Operaciones Morfológicas y CC =====================

// Se mantienen las funciones de ayuda necesarias para la compilación
static Mat morphOpenK(const Mat& bin, int k) {
  Mat out;
  morphologyEx(bin, out, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphCloseK(const Mat& bin, int k) {
  Mat out;
  morphologyEx(bin, out, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphErodeK(const Mat& bin, int k) {
  Mat out;
  erode(bin, out, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat morphDilateK(const Mat& bin, int k) {
  Mat out;
  dilate(bin, out, getStructuringElement(MORPH_ELLIPSE, Size(k, k)));
  return out;
}

static Mat keepLargeCC(const Mat& bin, int minArea) {
  CV_Assert(bin.type() == CV_8U);
  Mat labels, stats, cents;
  int n = connectedComponentsWithStats(bin, labels, stats, cents, 8, CV_32S);
  Mat out = Mat::zeros(bin.size(), CV_8U);
  for (int i = 1; i < n; ++i) { 
    if (stats.at<int>(i, CC_STAT_AREA) >= minArea)
      out.setTo(255, labels == i);
  }
  return out;
}

static void fillHoles(Mat& bin) {
  CV_Assert(bin.type() == CV_8U);
  Mat inv; bitwise_not(bin, inv);
  Mat ffMask = Mat::zeros(inv.rows + 2, inv.cols + 2, CV_8U);
  Mat filled = inv.clone();
  floodFill(filled, ffMask, Point(0, 0), Scalar(0));
  Mat holes;
  subtract(inv, filled, holes); 
  bin |= holes;                 
}

// ===================== 1. Segmentación Anatómica en HU reales =====================
AnatomyMasks generateAnatomicalMasksHU(const Mat& hu32f) {
  CV_Assert(!hu32f.empty() && hu32f.type() == CV_32F);
  AnatomyMasks M;

  // 1) Segmentación del Cuerpo
  Mat body = (hu32f > -300.f);
  body.convertTo(body, CV_8U, 255);
  body = morphCloseK(body, 7);
  fillHoles(body);

  if (countNonZero(body) < 1000) {
    M.fat = M.muscle_tendon = M.bones = Mat::zeros(hu32f.size(), CV_8U);
    return M;
  }

  // 2) Umbralización por Rangos HU
  Mat fat, muscle, bone;
  // Usamos lógica booleana para asegurar la consistencia con el código de debug de main.cpp
  fat    = (hu32f >= -250.f) & (hu32f <= -50.f);      // Grasa: -250 a -50 HU
  muscle = (hu32f > -50.f) & (hu32f <= 150.f);    // Músculo/Tejido Blando: -50 a 150 HU
  bone   = (hu32f > 500.f);     // Hueso Cortical: 500+ HU (ULTRA ESTRICTO)

  fat.convertTo(fat, CV_8U);
  muscle.convertTo(muscle, CV_8U);
  bone.convertTo(bone, CV_8U);

  // 3) Recorte a cuerpo
  fat    &= body;
  muscle &= body;
  bone   &= body;

  // 5) Limpieza Morfológica MÍNIMA y Componentes Conexos (Filtros ultra-suaves)
  bone   = morphCloseK(bone, 3);

  // MinArea muy bajos para preservar las regiones grandes que puedan estar fragmentadas
  fat    = keepLargeCC(fat,     5); 
  muscle = keepLargeCC(muscle, 10); 
  bone   = keepLargeCC(bone,     50);

  fillHoles(muscle); 
  // AJUSTE CRÍTICO: NO RELLENAR LOS HUECOS DEL HUESO (Médula Ósea)
  // fillHoles(bone); // <-- COMENTADO O ELIMINADO

  // 6) Prioridad (Hueso > Músculo > Grasa)
  
  // Prioridad Músculo > Grasa 
  fat.setTo(0, muscle); 

  // Prioridad Hueso > Músculo y Grasa
  // Usamos EROSIÓN MÁS AGRESIVA (K=3) en el hueso para proteger el tejido blando circundante.
  Mat bone_erode = morphErodeK(bone, 3); 
  
  muscle.setTo(0, bone_erode); 
  fat.setTo(0, bone_erode);    

  M.fat            = fat;
  M.muscle_tendon  = muscle;
  M.bones          = bone;
  return M;
}

// ===================== 2. Highlight ROI (Función de ejemplo) =====================
HighlightResult highlightROI(const Mat& gray) {
  // Se mantiene la función de ejemplo
  HighlightResult R;
  CV_Assert(!gray.empty());

  Mat g;
  if (gray.channels() == 3) cvtColor(gray, g, COLOR_BGR2GRAY);
  else g = gray.clone();

  threshold(g, R.mask, 28, 255, THRESH_BINARY); 
  R.mask = morphCloseK(R.mask, 9);
  fillHoles(R.mask);

  Mat baseBGR; cvtColor(g, baseBGR, COLOR_GRAY2BGR);
  R.overlay = baseBGR.clone();
  Mat color = Mat::zeros(baseBGR.size(), baseBGR.type());
  color.setTo(Scalar(0, 0, 255), R.mask); 
  addWeighted(R.overlay, 0.6, color, 0.4, 0, R.finalVis);

  return R;
}


// ===================== 3. Overlay coloreado (Alto Contraste) =====================
Mat colorizeAndOverlay(const Mat& slice8u, const AnatomyMasks& m) {
  Mat base;
  if (slice8u.channels() == 1) cvtColor(slice8u, base, COLOR_GRAY2BGR);
  else base = slice8u.clone();

  // Definición de colores BGR (Alto Contraste)
  Mat color = Mat::zeros(base.size(), base.type());
  
  // 1. Grasa (Amarillo)
  color.setTo(Scalar(  0, 255, 255), m.fat);           
  // 2. Músculo/Tendón (Rojo)
  color.setTo(Scalar(  0,   0, 255), m.muscle_tendon); 
  // 3. Hueso (Azul)
  color.setTo(Scalar(255,   0,   0), m.bones);         
  
  Mat out;
  // Resaltado fuerte (70% de máscara, 30% de fondo)
  addWeighted(base, 0.3, color, 0.7, 0, out);

  // Dibujo de Contornos (Opcional, pero útil para visualización)
  auto drawContour = [&](const Mat& mask, const Scalar& bgr) {
    Mat edges; Canny(mask, edges, 50, 150);
    Mat dil; dilate(edges, dil, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    out.setTo(bgr, dil);
  };
  drawContour(m.fat,           Scalar(  0, 150, 150));
  drawContour(m.muscle_tendon, Scalar(  0,   0, 150));
  drawContour(m.bones,         Scalar(150,   0,   0));

  return out;
}