// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x0000 : Control signals
//          bit 0  - ap_start (Read/Write/COH)
//          bit 1  - ap_done (Read/COR)
//          bit 2  - ap_idle (Read)
//          bit 3  - ap_ready (Read)
//          bit 7  - auto_restart (Read/Write)
//          others - reserved
// 0x0004 : Global Interrupt Enable Register
//          bit 0  - Global Interrupt Enable (Read/Write)
//          others - reserved
// 0x0008 : IP Interrupt Enable Register (Read/Write)
//          bit 0  - enable ap_done interrupt (Read/Write)
//          bit 1  - enable ap_ready interrupt (Read/Write)
//          others - reserved
// 0x000c : IP Interrupt Status Register (Read/TOW)
//          bit 0  - ap_done (COR/TOW)
//          bit 1  - ap_ready (COR/TOW)
//          others - reserved
// 0x0010 : Data signal of ap_return
//          bit 31~0 - ap_return[31:0] (Read)
// 0x2000 ~
// 0x3fff : Memory 'input_r' (1536 * 32b)
//          Word n : bit [31:0] - input_r[n]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XRUNNER_CONTROL_ADDR_AP_CTRL      0x0000
#define XRUNNER_CONTROL_ADDR_GIE          0x0004
#define XRUNNER_CONTROL_ADDR_IER          0x0008
#define XRUNNER_CONTROL_ADDR_ISR          0x000c
#define XRUNNER_CONTROL_ADDR_AP_RETURN    0x0010
#define XRUNNER_CONTROL_BITS_AP_RETURN    32
#define XRUNNER_CONTROL_ADDR_INPUT_R_BASE 0x2000
#define XRUNNER_CONTROL_ADDR_INPUT_R_HIGH 0x3fff
#define XRUNNER_CONTROL_WIDTH_INPUT_R     32
#define XRUNNER_CONTROL_DEPTH_INPUT_R     1536

