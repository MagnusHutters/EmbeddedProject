// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xrunner.h"

extern XRunner_Config XRunner_ConfigTable[];

XRunner_Config *XRunner_LookupConfig(u16 DeviceId) {
	XRunner_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XRUNNER_NUM_INSTANCES; Index++) {
		if (XRunner_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRunner_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRunner_Initialize(XRunner *InstancePtr, u16 DeviceId) {
	XRunner_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRunner_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRunner_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

