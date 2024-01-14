/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include <xil_io.h>
#include <xuartps.h>
#include "xparameters.h"
#include "sleep.h"
#include "xrunner.h"

void sendU32(XUartPs *uart, u32 data) {
    // Send each byte of the u32 data
    for (int i = 0; i < sizeof(u32); i++) {
        XUartPs_SendByte(uart->Config.BaseAddress, (data >> (8 * i)) & 0xFF);
    }
}

#define DATA_SIZE 1000
u32 j=0;
u32 imageSize= 48*32;
u32 image[1536];
u8 bytesRead;
u32 output=0;
XRunner nn;
XRunner_Config *nn_config;


int main()
{
    init_platform();
    xil_printf("Hello World\n\r");
    XUartPs UART;
    XUartPs_Config *Config;
    u8 ReceiveBuffer[1536];
    nn_config=XRunner_LookupConfig(XPAR_RUNNER_0_DEVICE_ID);
	XRunner_CfgInitialize(&nn,nn_config);
	XRunner_Initialize(&nn,XPAR_RUNNER_0_DEVICE_ID);
	//xil_printf("output");
    // Initialize UART
    Config = XUartPs_LookupConfig(XPAR_PSU_UART_1_DEVICE_ID);
    XUartPs_CfgInitialize(&UART, Config, Config->BaseAddress);

    while (1) {


        // Read data from UART
        bytesRead = XUartPs_Recv(&UART, ReceiveBuffer, sizeof(ReceiveBuffer));
        if(bytesRead > 0){
        	// Process or display received data
        	image [j] = *ReceiveBuffer;
        	if (j<imageSize){
        		//sendU32(&UART,j);
        		j ++;
        	}
        }
    	//XUartPs_SendByte(Config->BaseAddress,'5');
        if(j==imageSize){
        	XRunner_Write_input_r_Words(&nn,0,image,1536);
        	XRunner_Start(&nn);
        	while(!XRunner_IsDone(&nn));
        	output= XRunner_Get_return(&nn);
        	//sendU32(&UART,5);
        	sendU32(&UART,output);
        	j=0;
        }


    }


    return 0;

}

