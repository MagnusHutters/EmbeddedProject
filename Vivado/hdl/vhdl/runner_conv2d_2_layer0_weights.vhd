-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity runner_conv2d_2_layer0_weights_rom is 
    generic(
             DWIDTH     : integer := 32; 
             AWIDTH     : integer := 6; 
             MEM_SIZE    : integer := 54
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of runner_conv2d_2_layer0_weights_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "10111100110010111101101110111101", 
    1 => "10111110000111010101010000110111", 
    2 => "00111110100000101110101001011110", 
    3 => "00111100101110100001111001010001", 
    4 => "10111110001001010101101101011001", 
    5 => "00111111000001010111101000010101", 
    6 => "00111110100000101100101101000110", 
    7 => "10111110011100110001101001010000", 
    8 => "10111110001010010001100000111001", 
    9 => "00111110011101001100011111010010", 
    10 => "00111110100111111000101100001101", 
    11 => "00111111000011001110101101001010", 
    12 => "00111110001101010001101100000100", 
    13 => "10111110010010110011110111100011", 
    14 => "00111110111111100011011011111001", 
    15 => "10111110100000100001011010000000", 
    16 => "10111110011011111110100001010011", 
    17 => "00111101100001011011010101000011", 
    18 => "00111111000101100001011010001100", 
    19 => "00111110101100101010111010010000", 
    20 => "10111110100001100111100000110100", 
    21 => "00111110000111011001001010101111", 
    22 => "10111110101001111010101001011111", 
    23 => "00111110010110011000111111110011", 
    24 => "00111111000001011101000011011111", 
    25 => "00111101110001111011100011001001", 
    26 => "00111110010100010010110010110011", 
    27 => "10111110100111101101010100110101", 
    28 => "00111101001010001000000001111000", 
    29 => "10111101000000010100001010001111", 
    30 => "00111110110100010110001101000010", 
    31 => "00111100100110001110100110110101", 
    32 => "00111110111100011110110011000100", 
    33 => "00111100110101100100001010111100", 
    34 => "10111110000000000111110100110100", 
    35 => "00111110111100110011001000101111", 
    36 => "00111110110110100110101111111100", 
    37 => "00111110111000101000111100100110", 
    38 => "10111110010101010000010110011101", 
    39 => "10111110101001110101110101111111", 
    40 => "00111101111001101101000101011111", 
    41 => "00111011000110011111110111111011", 
    42 => "00111110101000000101111100000110", 
    43 => "00111100001111010001001100010110", 
    44 => "00111110101100110111110011000001", 
    45 => "10111101100111000010000100010101", 
    46 => "00111110110111000000010010100110", 
    47 => "00111110110100010111100111001110", 
    48 => "00111111000100001100111110011001", 
    49 => "10111101111100011111111001011111", 
    50 => "00111110101001000001110010000100", 
    51 => "10111110000001010011011001001111", 
    52 => "00111100110110111111100011001010", 
    53 => "00111110110000000010101111101010" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity runner_conv2d_2_layer0_weights is
    generic (
        DataWidth : INTEGER := 32;
        AddressRange : INTEGER := 54;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of runner_conv2d_2_layer0_weights is
    component runner_conv2d_2_layer0_weights_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    runner_conv2d_2_layer0_weights_rom_U :  component runner_conv2d_2_layer0_weights_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


