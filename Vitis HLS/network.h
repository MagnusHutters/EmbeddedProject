

#pragma once



void dense_layer3(float* input_array, const float* weights, const int inputsNum, const int outputsNum, float* output,const int doSoftmax);

void conv2d(int kernelI, int kernelJ, int inputX, int inputY, int inputDepth,
            int outputX, int outputY, int outputDepth,
            float* weights, float* inputs, float* output);

void maxPooling2D(float* input_feature_map, int input_height, int input_width, int depth,
                  float* output_feature_map, int output_height, int output_width);
