// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XRUNNER_H
#define XRUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xrunner_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u32 Control_BaseAddress;
} XRunner_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XRunner;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRunner_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRunner_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRunner_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRunner_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XRunner_Initialize(XRunner *InstancePtr, u16 DeviceId);
XRunner_Config* XRunner_LookupConfig(u16 DeviceId);
int XRunner_CfgInitialize(XRunner *InstancePtr, XRunner_Config *ConfigPtr);
#else
int XRunner_Initialize(XRunner *InstancePtr, const char* InstanceName);
int XRunner_Release(XRunner *InstancePtr);
#endif

void XRunner_Start(XRunner *InstancePtr);
u32 XRunner_IsDone(XRunner *InstancePtr);
u32 XRunner_IsIdle(XRunner *InstancePtr);
u32 XRunner_IsReady(XRunner *InstancePtr);
void XRunner_EnableAutoRestart(XRunner *InstancePtr);
void XRunner_DisableAutoRestart(XRunner *InstancePtr);
u32 XRunner_Get_return(XRunner *InstancePtr);

u32 XRunner_Get_input_r_BaseAddress(XRunner *InstancePtr);
u32 XRunner_Get_input_r_HighAddress(XRunner *InstancePtr);
u32 XRunner_Get_input_r_TotalBytes(XRunner *InstancePtr);
u32 XRunner_Get_input_r_BitWidth(XRunner *InstancePtr);
u32 XRunner_Get_input_r_Depth(XRunner *InstancePtr);
u32 XRunner_Write_input_r_Words(XRunner *InstancePtr, int offset, word_type *data, int length);
u32 XRunner_Read_input_r_Words(XRunner *InstancePtr, int offset, word_type *data, int length);
u32 XRunner_Write_input_r_Bytes(XRunner *InstancePtr, int offset, char *data, int length);
u32 XRunner_Read_input_r_Bytes(XRunner *InstancePtr, int offset, char *data, int length);

void XRunner_InterruptGlobalEnable(XRunner *InstancePtr);
void XRunner_InterruptGlobalDisable(XRunner *InstancePtr);
void XRunner_InterruptEnable(XRunner *InstancePtr, u32 Mask);
void XRunner_InterruptDisable(XRunner *InstancePtr, u32 Mask);
void XRunner_InterruptClear(XRunner *InstancePtr, u32 Mask);
u32 XRunner_InterruptGetEnabled(XRunner *InstancePtr);
u32 XRunner_InterruptGetStatus(XRunner *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
