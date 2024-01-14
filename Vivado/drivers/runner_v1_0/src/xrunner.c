// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xrunner.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRunner_CfgInitialize(XRunner *InstancePtr, XRunner_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRunner_Start(XRunner *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRunner_IsDone(XRunner *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRunner_IsIdle(XRunner *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRunner_IsReady(XRunner *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRunner_EnableAutoRestart(XRunner *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRunner_DisableAutoRestart(XRunner *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_CTRL, 0);
}

u32 XRunner_Get_return(XRunner *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_AP_RETURN);
    return Data;
}
u32 XRunner_Get_input_r_BaseAddress(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_BASE);
}

u32 XRunner_Get_input_r_HighAddress(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_HIGH);
}

u32 XRunner_Get_input_r_TotalBytes(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (XRUNNER_CONTROL_ADDR_INPUT_R_HIGH - XRUNNER_CONTROL_ADDR_INPUT_R_BASE + 1);
}

u32 XRunner_Get_input_r_BitWidth(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRUNNER_CONTROL_WIDTH_INPUT_R;
}

u32 XRunner_Get_input_r_Depth(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRUNNER_CONTROL_DEPTH_INPUT_R;
}

u32 XRunner_Write_input_r_Words(XRunner *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XRUNNER_CONTROL_ADDR_INPUT_R_HIGH - XRUNNER_CONTROL_ADDR_INPUT_R_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(int *)(InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_BASE + (offset + i)*4) = *(data + i);
    }
    return length;
}

u32 XRunner_Read_input_r_Words(XRunner *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XRUNNER_CONTROL_ADDR_INPUT_R_HIGH - XRUNNER_CONTROL_ADDR_INPUT_R_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(int *)(InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_BASE + (offset + i)*4);
    }
    return length;
}

u32 XRunner_Write_input_r_Bytes(XRunner *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XRUNNER_CONTROL_ADDR_INPUT_R_HIGH - XRUNNER_CONTROL_ADDR_INPUT_R_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(char *)(InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_BASE + offset + i) = *(data + i);
    }
    return length;
}

u32 XRunner_Read_input_r_Bytes(XRunner *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XRUNNER_CONTROL_ADDR_INPUT_R_HIGH - XRUNNER_CONTROL_ADDR_INPUT_R_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(char *)(InstancePtr->Control_BaseAddress + XRUNNER_CONTROL_ADDR_INPUT_R_BASE + offset + i);
    }
    return length;
}

void XRunner_InterruptGlobalEnable(XRunner *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_GIE, 1);
}

void XRunner_InterruptGlobalDisable(XRunner *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_GIE, 0);
}

void XRunner_InterruptEnable(XRunner *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_IER);
    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_IER, Register | Mask);
}

void XRunner_InterruptDisable(XRunner *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_IER);
    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRunner_InterruptClear(XRunner *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRunner_WriteReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_ISR, Mask);
}

u32 XRunner_InterruptGetEnabled(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_IER);
}

u32 XRunner_InterruptGetStatus(XRunner *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRunner_ReadReg(InstancePtr->Control_BaseAddress, XRUNNER_CONTROL_ADDR_ISR);
}

