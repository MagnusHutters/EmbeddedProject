


//#include "denseImplementation.h"
#include <ap_int.h>


#include <stdio.h>
#include <math.h>

#include "network.h"
#include "weights.h"

void dense_layer3(float* input_array, const float* weights, const int inputsNum, const int outputsNum, float* output,const int doSoftmax) {
    // Initialize output array to 0
    for (int i = 0; i < outputsNum; i++) {
        output[i] = 0.0f;
    }

    // Matrix multiplication
    for (int i = 0; i < inputsNum; i++) {

        for (int j = 0; j < outputsNum; j++) {
        	//printf("i: %d, j: %d\n", i, j);
            output[j] += input_array[i] * weights[i*outputsNum + j];
        }
    }

    //applying ReLU or Softmax activation
    if(doSoftmax==0){
        for (int i = 0; i < outputsNum; i++) {
            if (output[i] < 0.0f) {
                output[i] = 0.0f;
            }
        }
    }
    else{
        double max = output[0];
        for (int i = 1; i < outputsNum; i++) {
            if (output[i] > max) {
                max = output[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < outputsNum; i++) {
            output[i] = expf(output[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < outputsNum; i++) {
            output[i] /= sum;
        }

    }
}


void conv2d(int kernelI, int kernelJ, int inputX, int inputY, int inputDepth,
            int outputX, int outputY, int outputDepth,
            const float* weights, float* inputs, float* output) {

    //printf("kernelI: %d, kernelJ: %d, inputX: %d, inputY: %d, inputDepth: %d, outputX: %d, outputY: %d, outputDepth: %d\n", kernelI, kernelJ, inputX, inputY, inputDepth, outputX, outputY, outputDepth);
    int c = 0;

    for (int x = 0; x < outputX; x++) {
        for (int y = 0; y < outputY; y++) {
            for (int z = 0; z < outputDepth; z++) {
                float sum = 0;
                //c++;
                for (int i = 0; i < kernelI; i++) {
                    for (int j = 0; j < kernelJ; j++) {
                        for (int k = 0; k < inputDepth; k++) {
                            int weightIndex = i * (kernelJ * inputDepth * outputDepth)
                                            + j * (inputDepth * outputDepth)
                                            + k * outputDepth
                                            + z;
                            float weight = weights[weightIndex];

                            int inputIndex = 0//k * (inputX * inputY * inputDepth)
                                           + (x + i) * (inputY * inputDepth)
                                           + (y + j) * inputDepth
                                           + k;
                            float inputTemp = inputs[inputIndex];

                            //print("x: ", x, " y: ", y, " z: ", z, " i: ", i, " j: ", j, " k: ", k, " weight: ", weight, " inputTemp: ", inputTemp)

                            //printf("x: %d, y: %d, z: %d, i: %d, j: %d, k: %d, weight: %f, inputTemp: %f\n", x, y, z, i, j, k, weight, inputTemp);

                            sum += weight * inputTemp;
                        }
                    }
                }


                //printf("x: %d, y: %d, z: %d, sum: %f\n", x, y, z, sum);
                //if (c == 10) {
                //    return;
                //}

                int outputIndex = x * (outputY * outputDepth)
                                + y * outputDepth
                                + z;
                output[outputIndex] = sum;
            }
        }
    }
    // Apply ReLU activation

    for(int i=0;i<outputX*outputY*outputDepth;i++){
    	if(output[i]<0){
    		output[i]=0;
    	}
    }
}


void maxPooling2D(float* input_feature_map, int input_height, int input_width, int depth,
                  float* output_feature_map, int output_height, int output_width) {
    int pool_size = 2;
    int stride = 2;
#pragma hls unroll
    for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
            for (int d = 0; d < depth; d++) {
                float max_val = -__FLT_MAX__;

                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        // Expanded input index calculation
                        int input_index = (\
                            (h * stride + i) * input_width * depth + \
                            (w * stride + j) * depth + \
                            d \
                        );
                        max_val = (max_val > input_feature_map[input_index]) ? max_val : input_feature_map[input_index];
                    }
                }

                // Expanded output index calculation
                int output_index = (\
                    h * output_width * depth + \
                    w * depth + \
                    d \
                );
                output_feature_map[output_index] = max_val;
            }
        }
    }
}


float layer_0_output[layer0_totalOutputSize];
float layer_1_output[layer1_totalOutputSize];
float layer_2_output[layer2_totalOutputSize];
float layer_3_output[layer3_totalOutputSize];
float layer_4_output[layer4_totalOutputSize];
float layer_5_output[layer5_totalOutputSize];
float layer_6_output[layer6_totalOutputSize];
float layer_7_output[layer7_totalOutputSize];
float layer_8_output[layer8_totalOutputSize];

int runner(int input[1536]){
#pragma hls pipeline II =1

	#pragma hls interface s_axilite port=input
	#pragma hls interface s_axilite port=return
	float inputImage[1536]={};

	for(int i =0; i< 1536; i++)
	{
#pragma HLS unroll
		inputImage[i]=input[i]/255.0;
	}


    conv2d(layer0_kernelSizeX, layer0_kernelSizeY, layer0_inputX, layer0_inputY, layer0_inputDepth,
           layer0_outputX, layer0_outputY, layer0_outputDepth,
           layer0_weights, inputImage, layer_0_output);
    //return;

    maxPooling2D(layer_0_output, layer1_inputX, layer1_inputY, layer1_inputDepth,
                 layer_1_output, layer1_outputX, layer1_outputY);

    conv2d(layer2_kernelSizeX, layer2_kernelSizeY, layer2_inputX, layer2_inputY, layer2_inputDepth,
              layer2_outputX, layer2_outputY, layer2_outputDepth,
              layer2_weights, layer_1_output, layer_2_output);
    maxPooling2D(layer_2_output, layer3_inputX, layer3_inputY, layer3_inputDepth,
                 layer_3_output, layer3_outputX, layer3_outputY);

    conv2d(layer4_kernelSizeX, layer4_kernelSizeY, layer4_inputX, layer4_inputY, layer4_inputDepth,
                layer4_outputX, layer4_outputY, layer4_outputDepth,
                layer4_weights, layer_3_output, layer_4_output);
    maxPooling2D(layer_4_output, layer5_inputX, layer5_inputY, layer5_inputDepth,
                 layer_5_output, layer5_outputX, layer5_outputY);

    //skip flatten layer - c array is already flattened

    dense_layer3(layer_5_output, layer7_weights, layer7_inputSize, layer7_outputSize, layer_7_output,0);

    dense_layer3(layer_7_output, layer8_weights, layer8_inputSize, layer8_outputSize, layer_8_output, 1);

    //Print 10 first values of layer output

    //printf("===== Layer output =====\n");
    for (int i = 0; i < 3; i++) {
        //printf("%f\n", layer_8_output[i]);
    }

    //printf("========================\n");


    for (int i = 0; i < 3; i++) {
    	if (round(layer_8_output[i]) == 1 ){
    		return i;
    	}
    }
    return -1;

}
