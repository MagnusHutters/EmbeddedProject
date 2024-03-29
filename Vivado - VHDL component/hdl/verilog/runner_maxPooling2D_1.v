// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Version: 2020.2
// Copyright (C) Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module runner_maxPooling2D_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        layer_2_output_address0,
        layer_2_output_ce0,
        layer_2_output_q0,
        layer_3_output_address0,
        layer_3_output_ce0,
        layer_3_output_we0,
        layer_3_output_d0,
        grp_fu_29505_p_din0,
        grp_fu_29505_p_din1,
        grp_fu_29505_p_opcode,
        grp_fu_29505_p_dout0,
        grp_fu_29505_p_ce
);

parameter    ap_ST_fsm_state1 = 11'd1;
parameter    ap_ST_fsm_state2 = 11'd2;
parameter    ap_ST_fsm_state3 = 11'd4;
parameter    ap_ST_fsm_state4 = 11'd8;
parameter    ap_ST_fsm_state5 = 11'd16;
parameter    ap_ST_fsm_state6 = 11'd32;
parameter    ap_ST_fsm_state7 = 11'd64;
parameter    ap_ST_fsm_state8 = 11'd128;
parameter    ap_ST_fsm_state9 = 11'd256;
parameter    ap_ST_fsm_state10 = 11'd512;
parameter    ap_ST_fsm_state11 = 11'd1024;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [11:0] layer_2_output_address0;
output   layer_2_output_ce0;
input  [31:0] layer_2_output_q0;
output  [9:0] layer_3_output_address0;
output   layer_3_output_ce0;
output   layer_3_output_we0;
output  [31:0] layer_3_output_d0;
output  [31:0] grp_fu_29505_p_din0;
output  [31:0] grp_fu_29505_p_din1;
output  [4:0] grp_fu_29505_p_opcode;
input  [0:0] grp_fu_29505_p_dout0;
output   grp_fu_29505_p_ce;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg[11:0] layer_2_output_address0;
reg layer_2_output_ce0;
reg layer_3_output_ce0;
reg layer_3_output_we0;

(* fsm_encoding = "none" *) reg   [10:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [31:0] reg_180;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state8;
wire    ap_CS_fsm_state10;
wire   [3:0] add_ln123_fu_186_p2;
reg   [3:0] add_ln123_reg_836;
wire    ap_CS_fsm_state2;
wire   [7:0] add_ln123_1_fu_192_p2;
reg   [7:0] add_ln123_1_reg_841;
wire   [7:0] empty_39_fu_228_p2;
reg   [7:0] empty_39_reg_849;
wire   [0:0] icmp_ln123_fu_198_p2;
wire   [8:0] empty_41_fu_244_p2;
reg   [8:0] empty_41_reg_854;
wire   [2:0] add_ln124_fu_250_p2;
reg   [2:0] add_ln124_reg_859;
wire    ap_CS_fsm_state3;
wire   [9:0] empty_45_fu_307_p2;
reg   [9:0] empty_45_reg_867;
wire   [0:0] icmp_ln124_fu_260_p2;
wire   [11:0] sub_ln131_fu_339_p2;
reg   [11:0] sub_ln131_reg_872;
wire   [11:0] sub_ln131_4_fu_371_p2;
reg   [11:0] sub_ln131_4_reg_877;
wire   [11:0] sub_ln131_5_fu_406_p2;
reg   [11:0] sub_ln131_5_reg_882;
wire   [11:0] sub_ln131_6_fu_442_p2;
reg   [11:0] sub_ln131_6_reg_887;
wire   [3:0] add_ln125_fu_448_p2;
reg   [3:0] add_ln125_reg_892;
wire    ap_CS_fsm_state4;
wire   [0:0] icmp_ln125_fu_462_p2;
wire   [11:0] add_ln136_4_fu_478_p2;
reg   [11:0] add_ln136_4_reg_905;
wire   [11:0] add_ln136_5_fu_483_p2;
reg   [11:0] add_ln136_5_reg_910;
wire   [11:0] add_ln136_6_fu_488_p2;
reg   [11:0] add_ln136_6_reg_915;
wire   [9:0] add_ln146_fu_493_p2;
reg   [9:0] add_ln146_reg_920;
wire   [31:0] select_ln136_fu_544_p3;
reg   [31:0] select_ln136_reg_930;
wire   [31:0] select_ln136_4_fu_636_p3;
reg   [31:0] select_ln136_4_reg_937;
wire    ap_CS_fsm_state7;
wire   [31:0] select_ln136_5_fu_730_p3;
reg   [31:0] select_ln136_5_reg_949;
wire    ap_CS_fsm_state9;
reg   [3:0] h_reg_128;
reg   [7:0] phi_mul_reg_139;
reg   [2:0] w_reg_151;
reg   [3:0] d_reg_162;
wire    ap_CS_fsm_state11;
wire   [63:0] zext_ln136_fu_473_p1;
wire   [63:0] zext_ln136_4_fu_498_p1;
wire   [63:0] zext_ln136_5_fu_643_p1;
wire   [63:0] zext_ln136_6_fu_737_p1;
wire   [63:0] zext_ln146_fu_832_p1;
reg   [31:0] grp_fu_173_p0;
reg   [31:0] grp_fu_173_p1;
wire   [4:0] tmp_21_fu_204_p3;
wire   [6:0] p_shl_fu_216_p3;
wire   [7:0] p_shl_cast_fu_224_p1;
wire   [7:0] p_cast_fu_212_p1;
wire   [4:0] empty_40_fu_234_p2;
wire   [4:0] empty_41_fu_244_p0;
wire   [4:0] empty_41_fu_244_p1;
wire   [3:0] tmp_22_fu_266_p3;
wire   [7:0] zext_ln124_fu_256_p1;
wire   [7:0] empty_43_fu_282_p2;
wire   [5:0] empty_44_fu_287_p1;
wire   [9:0] p_shl2_fu_291_p3;
wire   [9:0] p_shl3_fu_299_p3;
wire   [7:0] p_cast6_fu_278_p1;
wire   [7:0] add_ln131_fu_313_p2;
wire   [9:0] shl_ln131_8_fu_327_p3;
wire   [11:0] shl_ln_fu_319_p3;
wire   [11:0] zext_ln131_fu_335_p1;
wire   [7:0] or_ln131_fu_345_p2;
wire   [9:0] shl_ln131_s_fu_359_p3;
wire   [11:0] shl_ln131_9_fu_351_p3;
wire   [11:0] zext_ln131_1_fu_367_p1;
wire   [8:0] p_cast9_fu_274_p1;
wire   [8:0] add_ln131_2_fu_377_p2;
wire   [7:0] trunc_ln131_fu_382_p1;
wire   [10:0] shl_ln131_2_fu_394_p3;
wire   [11:0] shl_ln131_1_fu_386_p3;
wire   [11:0] zext_ln131_2_fu_402_p1;
wire   [8:0] add_ln131_3_fu_412_p2;
wire   [7:0] trunc_ln131_4_fu_418_p1;
wire   [10:0] shl_ln131_4_fu_430_p3;
wire   [11:0] shl_ln131_3_fu_422_p3;
wire   [11:0] zext_ln131_3_fu_438_p1;
wire   [11:0] zext_ln125_2_fu_458_p1;
wire   [11:0] add_ln136_fu_468_p2;
wire   [9:0] zext_ln125_fu_454_p1;
wire   [31:0] bitcast_ln136_fu_502_p1;
wire   [7:0] tmp_fu_506_p4;
wire   [22:0] trunc_ln136_fu_516_p1;
wire   [0:0] icmp_ln136_14_fu_526_p2;
wire   [0:0] icmp_ln136_fu_520_p2;
wire   [0:0] or_ln136_fu_532_p2;
wire   [0:0] grp_fu_173_p2;
wire   [0:0] and_ln136_fu_538_p2;
wire   [31:0] bitcast_ln136_7_fu_553_p1;
wire   [31:0] bitcast_ln136_8_fu_570_p1;
wire   [7:0] tmp_12_fu_556_p4;
wire   [22:0] trunc_ln136_7_fu_566_p1;
wire   [0:0] icmp_ln136_16_fu_594_p2;
wire   [0:0] icmp_ln136_15_fu_588_p2;
wire   [7:0] tmp_13_fu_574_p4;
wire   [22:0] trunc_ln136_8_fu_584_p1;
wire   [0:0] icmp_ln136_18_fu_612_p2;
wire   [0:0] icmp_ln136_17_fu_606_p2;
wire   [0:0] or_ln136_7_fu_600_p2;
wire   [0:0] or_ln136_8_fu_618_p2;
wire   [0:0] and_ln136_7_fu_624_p2;
wire   [0:0] and_ln136_8_fu_630_p2;
wire   [31:0] bitcast_ln136_9_fu_647_p1;
wire   [31:0] bitcast_ln136_10_fu_664_p1;
wire   [7:0] tmp_15_fu_650_p4;
wire   [22:0] trunc_ln136_9_fu_660_p1;
wire   [0:0] icmp_ln136_20_fu_688_p2;
wire   [0:0] icmp_ln136_19_fu_682_p2;
wire   [7:0] tmp_16_fu_668_p4;
wire   [22:0] trunc_ln136_10_fu_678_p1;
wire   [0:0] icmp_ln136_22_fu_706_p2;
wire   [0:0] icmp_ln136_21_fu_700_p2;
wire   [0:0] or_ln136_9_fu_694_p2;
wire   [0:0] or_ln136_10_fu_712_p2;
wire   [0:0] and_ln136_9_fu_718_p2;
wire   [0:0] and_ln136_10_fu_724_p2;
wire   [31:0] bitcast_ln136_11_fu_741_p1;
wire   [31:0] bitcast_ln136_12_fu_758_p1;
wire   [7:0] tmp_18_fu_744_p4;
wire   [22:0] trunc_ln136_11_fu_754_p1;
wire   [0:0] icmp_ln136_24_fu_782_p2;
wire   [0:0] icmp_ln136_23_fu_776_p2;
wire   [7:0] tmp_19_fu_762_p4;
wire   [22:0] trunc_ln136_12_fu_772_p1;
wire   [0:0] icmp_ln136_26_fu_800_p2;
wire   [0:0] icmp_ln136_25_fu_794_p2;
wire   [0:0] or_ln136_11_fu_788_p2;
wire   [0:0] or_ln136_12_fu_806_p2;
wire   [0:0] and_ln136_11_fu_812_p2;
wire   [0:0] and_ln136_12_fu_818_p2;
wire    grp_fu_173_ce;
reg   [4:0] grp_fu_173_opcode;
reg   [10:0] ap_NS_fsm;
wire   [8:0] empty_41_fu_244_p00;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 11'd1;
end

runner_mul_5ns_5ns_9_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 5 ),
    .din1_WIDTH( 5 ),
    .dout_WIDTH( 9 ))
mul_5ns_5ns_9_1_1_U22(
    .din0(empty_41_fu_244_p0),
    .din1(empty_41_fu_244_p1),
    .dout(empty_41_fu_244_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln124_fu_260_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        d_reg_162 <= 4'd0;
    end else if ((1'b1 == ap_CS_fsm_state11)) begin
        d_reg_162 <= add_ln125_reg_892;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        h_reg_128 <= 4'd0;
    end else if (((icmp_ln124_fu_260_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        h_reg_128 <= add_ln123_reg_836;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        phi_mul_reg_139 <= 8'd0;
    end else if (((icmp_ln124_fu_260_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        phi_mul_reg_139 <= add_ln123_1_reg_841;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln123_fu_198_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        w_reg_151 <= 3'd0;
    end else if (((1'b1 == ap_CS_fsm_state4) & (icmp_ln125_fu_462_p2 == 1'd1))) begin
        w_reg_151 <= add_ln124_reg_859;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        add_ln123_1_reg_841 <= add_ln123_1_fu_192_p2;
        add_ln123_reg_836 <= add_ln123_fu_186_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        add_ln124_reg_859 <= add_ln124_fu_250_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        add_ln125_reg_892 <= add_ln125_fu_448_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state4) & (icmp_ln125_fu_462_p2 == 1'd0))) begin
        add_ln136_4_reg_905 <= add_ln136_4_fu_478_p2;
        add_ln136_5_reg_910 <= add_ln136_5_fu_483_p2;
        add_ln136_6_reg_915 <= add_ln136_6_fu_488_p2;
        add_ln146_reg_920 <= add_ln146_fu_493_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln123_fu_198_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        empty_39_reg_849[7 : 1] <= empty_39_fu_228_p2[7 : 1];
        empty_41_reg_854 <= empty_41_fu_244_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln124_fu_260_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        empty_45_reg_867[9 : 2] <= empty_45_fu_307_p2[9 : 2];
        sub_ln131_4_reg_877[11 : 3] <= sub_ln131_4_fu_371_p2[11 : 3];
        sub_ln131_5_reg_882[11 : 2] <= sub_ln131_5_fu_406_p2[11 : 2];
        sub_ln131_6_reg_887[11 : 2] <= sub_ln131_6_fu_442_p2[11 : 2];
        sub_ln131_reg_872[11 : 2] <= sub_ln131_fu_339_p2[11 : 2];
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state6) | (1'b1 == ap_CS_fsm_state5))) begin
        reg_180 <= layer_2_output_q0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        select_ln136_4_reg_937 <= select_ln136_4_fu_636_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        select_ln136_5_reg_949 <= select_ln136_5_fu_730_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        select_ln136_reg_930 <= select_ln136_fu_544_p3;
    end
end

always @ (*) begin
    if ((((icmp_ln123_fu_198_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln123_fu_198_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state6))) begin
        grp_fu_173_opcode = 5'd2;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        grp_fu_173_opcode = 5'd4;
    end else begin
        grp_fu_173_opcode = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        grp_fu_173_p0 = select_ln136_5_reg_949;
    end else if ((1'b1 == ap_CS_fsm_state8)) begin
        grp_fu_173_p0 = select_ln136_4_reg_937;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        grp_fu_173_p0 = select_ln136_fu_544_p3;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        grp_fu_173_p0 = layer_2_output_q0;
    end else begin
        grp_fu_173_p0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state6))) begin
        grp_fu_173_p1 = layer_2_output_q0;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        grp_fu_173_p1 = 32'd4286578687;
    end else begin
        grp_fu_173_p1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        layer_2_output_address0 = zext_ln136_6_fu_737_p1;
    end else if ((1'b1 == ap_CS_fsm_state7)) begin
        layer_2_output_address0 = zext_ln136_5_fu_643_p1;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        layer_2_output_address0 = zext_ln136_4_fu_498_p1;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        layer_2_output_address0 = zext_ln136_fu_473_p1;
    end else begin
        layer_2_output_address0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state5) | (1'b1 == ap_CS_fsm_state9) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state4))) begin
        layer_2_output_ce0 = 1'b1;
    end else begin
        layer_2_output_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        layer_3_output_ce0 = 1'b1;
    end else begin
        layer_3_output_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        layer_3_output_we0 = 1'b1;
    end else begin
        layer_3_output_we0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((icmp_ln123_fu_198_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((icmp_ln124_fu_260_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_CS_fsm_state4) & (icmp_ln125_fu_462_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln123_1_fu_192_p2 = (phi_mul_reg_139 + 8'd26);

assign add_ln123_fu_186_p2 = (h_reg_128 + 4'd1);

assign add_ln124_fu_250_p2 = (w_reg_151 + 3'd1);

assign add_ln125_fu_448_p2 = (d_reg_162 + 4'd1);

assign add_ln131_2_fu_377_p2 = (empty_41_reg_854 + p_cast9_fu_274_p1);

assign add_ln131_3_fu_412_p2 = (add_ln131_2_fu_377_p2 + 9'd1);

assign add_ln131_fu_313_p2 = (phi_mul_reg_139 + p_cast6_fu_278_p1);

assign add_ln136_4_fu_478_p2 = (sub_ln131_4_reg_877 + zext_ln125_2_fu_458_p1);

assign add_ln136_5_fu_483_p2 = (sub_ln131_5_reg_882 + zext_ln125_2_fu_458_p1);

assign add_ln136_6_fu_488_p2 = (sub_ln131_6_reg_887 + zext_ln125_2_fu_458_p1);

assign add_ln136_fu_468_p2 = (sub_ln131_reg_872 + zext_ln125_2_fu_458_p1);

assign add_ln146_fu_493_p2 = (zext_ln125_fu_454_p1 + empty_45_reg_867);

assign and_ln136_10_fu_724_p2 = (grp_fu_29505_p_dout0 & and_ln136_9_fu_718_p2);

assign and_ln136_11_fu_812_p2 = (or_ln136_12_fu_806_p2 & or_ln136_11_fu_788_p2);

assign and_ln136_12_fu_818_p2 = (grp_fu_29505_p_dout0 & and_ln136_11_fu_812_p2);

assign and_ln136_7_fu_624_p2 = (or_ln136_8_fu_618_p2 & or_ln136_7_fu_600_p2);

assign and_ln136_8_fu_630_p2 = (grp_fu_29505_p_dout0 & and_ln136_7_fu_624_p2);

assign and_ln136_9_fu_718_p2 = (or_ln136_9_fu_694_p2 & or_ln136_10_fu_712_p2);

assign and_ln136_fu_538_p2 = (or_ln136_fu_532_p2 & grp_fu_29505_p_dout0);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign bitcast_ln136_10_fu_664_p1 = reg_180;

assign bitcast_ln136_11_fu_741_p1 = select_ln136_5_reg_949;

assign bitcast_ln136_12_fu_758_p1 = reg_180;

assign bitcast_ln136_7_fu_553_p1 = select_ln136_reg_930;

assign bitcast_ln136_8_fu_570_p1 = reg_180;

assign bitcast_ln136_9_fu_647_p1 = select_ln136_4_reg_937;

assign bitcast_ln136_fu_502_p1 = reg_180;

assign empty_39_fu_228_p2 = (p_shl_cast_fu_224_p1 - p_cast_fu_212_p1);

assign empty_40_fu_234_p2 = (tmp_21_fu_204_p3 | 5'd1);

assign empty_41_fu_244_p0 = empty_41_fu_244_p00;

assign empty_41_fu_244_p00 = empty_40_fu_234_p2;

assign empty_41_fu_244_p1 = 9'd13;

assign empty_43_fu_282_p2 = (zext_ln124_fu_256_p1 + empty_39_reg_849);

assign empty_44_fu_287_p1 = empty_43_fu_282_p2[5:0];

assign empty_45_fu_307_p2 = (p_shl2_fu_291_p3 - p_shl3_fu_299_p3);

assign grp_fu_173_ce = 1'b1;

assign grp_fu_173_p2 = grp_fu_29505_p_dout0;

assign grp_fu_29505_p_ce = 1'b1;

assign grp_fu_29505_p_din0 = grp_fu_173_p0;

assign grp_fu_29505_p_din1 = grp_fu_173_p1;

assign grp_fu_29505_p_opcode = grp_fu_173_opcode;

assign icmp_ln123_fu_198_p2 = ((h_reg_128 == 4'd10) ? 1'b1 : 1'b0);

assign icmp_ln124_fu_260_p2 = ((w_reg_151 == 3'd6) ? 1'b1 : 1'b0);

assign icmp_ln125_fu_462_p2 = ((d_reg_162 == 4'd12) ? 1'b1 : 1'b0);

assign icmp_ln136_14_fu_526_p2 = ((trunc_ln136_fu_516_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_15_fu_588_p2 = ((tmp_12_fu_556_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_16_fu_594_p2 = ((trunc_ln136_7_fu_566_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_17_fu_606_p2 = ((tmp_13_fu_574_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_18_fu_612_p2 = ((trunc_ln136_8_fu_584_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_19_fu_682_p2 = ((tmp_15_fu_650_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_20_fu_688_p2 = ((trunc_ln136_9_fu_660_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_21_fu_700_p2 = ((tmp_16_fu_668_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_22_fu_706_p2 = ((trunc_ln136_10_fu_678_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_23_fu_776_p2 = ((tmp_18_fu_744_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_24_fu_782_p2 = ((trunc_ln136_11_fu_754_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_25_fu_794_p2 = ((tmp_19_fu_762_p4 != 8'd255) ? 1'b1 : 1'b0);

assign icmp_ln136_26_fu_800_p2 = ((trunc_ln136_12_fu_772_p1 == 23'd0) ? 1'b1 : 1'b0);

assign icmp_ln136_fu_520_p2 = ((tmp_fu_506_p4 != 8'd255) ? 1'b1 : 1'b0);

assign layer_3_output_address0 = zext_ln146_fu_832_p1;

assign layer_3_output_d0 = ((and_ln136_12_fu_818_p2[0:0] == 1'b1) ? select_ln136_5_reg_949 : reg_180);

assign or_ln131_fu_345_p2 = (8'd1 | add_ln131_fu_313_p2);

assign or_ln136_10_fu_712_p2 = (icmp_ln136_22_fu_706_p2 | icmp_ln136_21_fu_700_p2);

assign or_ln136_11_fu_788_p2 = (icmp_ln136_24_fu_782_p2 | icmp_ln136_23_fu_776_p2);

assign or_ln136_12_fu_806_p2 = (icmp_ln136_26_fu_800_p2 | icmp_ln136_25_fu_794_p2);

assign or_ln136_7_fu_600_p2 = (icmp_ln136_16_fu_594_p2 | icmp_ln136_15_fu_588_p2);

assign or_ln136_8_fu_618_p2 = (icmp_ln136_18_fu_612_p2 | icmp_ln136_17_fu_606_p2);

assign or_ln136_9_fu_694_p2 = (icmp_ln136_20_fu_688_p2 | icmp_ln136_19_fu_682_p2);

assign or_ln136_fu_532_p2 = (icmp_ln136_fu_520_p2 | icmp_ln136_14_fu_526_p2);

assign p_cast6_fu_278_p1 = tmp_22_fu_266_p3;

assign p_cast9_fu_274_p1 = tmp_22_fu_266_p3;

assign p_cast_fu_212_p1 = tmp_21_fu_204_p3;

assign p_shl2_fu_291_p3 = {{empty_44_fu_287_p1}, {4'd0}};

assign p_shl3_fu_299_p3 = {{empty_43_fu_282_p2}, {2'd0}};

assign p_shl_cast_fu_224_p1 = p_shl_fu_216_p3;

assign p_shl_fu_216_p3 = {{h_reg_128}, {3'd0}};

assign select_ln136_4_fu_636_p3 = ((and_ln136_8_fu_630_p2[0:0] == 1'b1) ? select_ln136_reg_930 : reg_180);

assign select_ln136_5_fu_730_p3 = ((and_ln136_10_fu_724_p2[0:0] == 1'b1) ? select_ln136_4_reg_937 : reg_180);

assign select_ln136_fu_544_p3 = ((and_ln136_fu_538_p2[0:0] == 1'b1) ? 32'd4286578687 : reg_180);

assign shl_ln131_1_fu_386_p3 = {{trunc_ln131_fu_382_p1}, {4'd0}};

assign shl_ln131_2_fu_394_p3 = {{add_ln131_2_fu_377_p2}, {2'd0}};

assign shl_ln131_3_fu_422_p3 = {{trunc_ln131_4_fu_418_p1}, {4'd0}};

assign shl_ln131_4_fu_430_p3 = {{add_ln131_3_fu_412_p2}, {2'd0}};

assign shl_ln131_8_fu_327_p3 = {{add_ln131_fu_313_p2}, {2'd0}};

assign shl_ln131_9_fu_351_p3 = {{or_ln131_fu_345_p2}, {4'd0}};

assign shl_ln131_s_fu_359_p3 = {{or_ln131_fu_345_p2}, {2'd0}};

assign shl_ln_fu_319_p3 = {{add_ln131_fu_313_p2}, {4'd0}};

assign sub_ln131_4_fu_371_p2 = (shl_ln131_9_fu_351_p3 - zext_ln131_1_fu_367_p1);

assign sub_ln131_5_fu_406_p2 = (shl_ln131_1_fu_386_p3 - zext_ln131_2_fu_402_p1);

assign sub_ln131_6_fu_442_p2 = (shl_ln131_3_fu_422_p3 - zext_ln131_3_fu_438_p1);

assign sub_ln131_fu_339_p2 = (shl_ln_fu_319_p3 - zext_ln131_fu_335_p1);

assign tmp_12_fu_556_p4 = {{bitcast_ln136_7_fu_553_p1[30:23]}};

assign tmp_13_fu_574_p4 = {{bitcast_ln136_8_fu_570_p1[30:23]}};

assign tmp_15_fu_650_p4 = {{bitcast_ln136_9_fu_647_p1[30:23]}};

assign tmp_16_fu_668_p4 = {{bitcast_ln136_10_fu_664_p1[30:23]}};

assign tmp_18_fu_744_p4 = {{bitcast_ln136_11_fu_741_p1[30:23]}};

assign tmp_19_fu_762_p4 = {{bitcast_ln136_12_fu_758_p1[30:23]}};

assign tmp_21_fu_204_p3 = {{h_reg_128}, {1'd0}};

assign tmp_22_fu_266_p3 = {{w_reg_151}, {1'd0}};

assign tmp_fu_506_p4 = {{bitcast_ln136_fu_502_p1[30:23]}};

assign trunc_ln131_4_fu_418_p1 = add_ln131_3_fu_412_p2[7:0];

assign trunc_ln131_fu_382_p1 = add_ln131_2_fu_377_p2[7:0];

assign trunc_ln136_10_fu_678_p1 = bitcast_ln136_10_fu_664_p1[22:0];

assign trunc_ln136_11_fu_754_p1 = bitcast_ln136_11_fu_741_p1[22:0];

assign trunc_ln136_12_fu_772_p1 = bitcast_ln136_12_fu_758_p1[22:0];

assign trunc_ln136_7_fu_566_p1 = bitcast_ln136_7_fu_553_p1[22:0];

assign trunc_ln136_8_fu_584_p1 = bitcast_ln136_8_fu_570_p1[22:0];

assign trunc_ln136_9_fu_660_p1 = bitcast_ln136_9_fu_647_p1[22:0];

assign trunc_ln136_fu_516_p1 = bitcast_ln136_fu_502_p1[22:0];

assign zext_ln124_fu_256_p1 = w_reg_151;

assign zext_ln125_2_fu_458_p1 = d_reg_162;

assign zext_ln125_fu_454_p1 = d_reg_162;

assign zext_ln131_1_fu_367_p1 = shl_ln131_s_fu_359_p3;

assign zext_ln131_2_fu_402_p1 = shl_ln131_2_fu_394_p3;

assign zext_ln131_3_fu_438_p1 = shl_ln131_4_fu_430_p3;

assign zext_ln131_fu_335_p1 = shl_ln131_8_fu_327_p3;

assign zext_ln136_4_fu_498_p1 = add_ln136_4_reg_905;

assign zext_ln136_5_fu_643_p1 = add_ln136_5_reg_910;

assign zext_ln136_6_fu_737_p1 = add_ln136_6_reg_915;

assign zext_ln136_fu_473_p1 = add_ln136_fu_468_p2;

assign zext_ln146_fu_832_p1 = add_ln146_reg_920;

always @ (posedge ap_clk) begin
    empty_39_reg_849[0] <= 1'b0;
    empty_45_reg_867[1:0] <= 2'b00;
    sub_ln131_reg_872[1:0] <= 2'b00;
    sub_ln131_4_reg_877[2:0] <= 3'b100;
    sub_ln131_5_reg_882[1:0] <= 2'b00;
    sub_ln131_6_reg_887[1:0] <= 2'b00;
end

endmodule //runner_maxPooling2D_1
