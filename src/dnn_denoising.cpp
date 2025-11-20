// src/dnn_denoising.cpp
#include "dnn_denoising.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Constructor
DnnDenoiser::DnnDenoiser(const std::string& modelPath) {
    // Intentar cargar la red. Si el archivo es incompatible, esto lanzará una excepción
    // que será atrapada en el main.
    net = dnn::readNetFromONNX(modelPath);
    
    if (net.empty()) {
        CV_Error(Error::StsError, "La red neuronal se cargó pero está vacía.");
    }
    
    // Configurar backend preferido (CPU es más seguro para compatibilidad)
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

// Método Denoise
Mat DnnDenoiser::denoise(const Mat& noisy8u) {
    if (net.empty()) return noisy8u.clone();
    
    // 1. Convertir a Float [0, 1]
    Mat input_mat;
    noisy8u.convertTo(input_mat, CV_32F, 1.0 / 255.0);

    // 2. Crear Blob (N, C, H, W)
    Mat blob = dnn::blobFromImage(input_mat); 

    // 3. Inferencia
    net.setInput(blob);
    Mat residual_blob = net.forward(); // La red DnCNN predice el RUIDO

    // 4. Postprocesamiento: Convertir blob a Mat 2D
    // El blob tiene dimensiones [1, 1, H, W]. Lo convertimos a [H, W]
    std::vector<int> sizes = {residual_blob.size[2], residual_blob.size[3]};
    Mat residual_mat(2, sizes.data(), CV_32F, residual_blob.ptr<float>());

    // 5. APRENDIZAJE RESIDUAL: Imagen Limpia = Entrada - Ruido Predicho
    Mat output_mat;
    subtract(input_mat, residual_mat, output_mat);

    // 6. Clamping y conversión a 8-bit
    output_mat.setTo(0.0f, output_mat < 0.0f);
    output_mat.setTo(1.0f, output_mat > 1.0f);
    
    Mat denoised8u;
    output_mat.convertTo(denoised8u, CV_8U, 255.0);

    return denoised8u;
}