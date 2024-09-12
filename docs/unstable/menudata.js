/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var menudata={children:[
{text:"Main Page",url:"index.html"},
{text:"Installation",url:"installation.html",children:[
{text:"Dependencies",url:"installation.html#dependencies"},
{text:"Installation steps",url:"installation.html#install_steps"},
{text:"Versions",url:"installation.html#versions"},
{text:"Custom CMake options",url:"installation.html#cmake_options"}]},
{text:"Integration in C++ projects",url:"integration.html",children:[
{text:"CMake",url:"integration.html#cmake",children:[
{text:"FetchContent",url:"integration.html#fetch"},
{text:"find_package",url:"integration.html#find_package"},
{text:"add_subdirectory",url:"integration.html#add_sub"}]},
{text:"Other",url:"integration.html#other"}]},
{text:"Examples",url:"examples.html",children:[
{text:"Compiling the examples",url:"examples.html#compiling"},
{text:"Example 1: Hello world!",url:"ex1.html"},
{text:"Example 2: Use monitor to communicate errors",url:"ex2.html"},
{text:"Example 3: Custom type and operator",url:"ex3.html"}]},
{text:"API Documentation",url:"documentation.html",children:[
{text:"MPI essentials",url:"group__mpi__essentials.html",children:[
{text:"communicator",url:"classmpi_1_1communicator.html"},
{text:"environment",url:"structmpi_1_1environment.html"},
{text:"is_initialized",url:"group__mpi__essentials.html#gaee54f343fdd8f1712ae521bd8ee69dfc"},
{text:"has_env",url:"group__mpi__essentials.html#ga590f450f6987d3e6c0398048515856b1"}]},
{text:"MPI datatypes and operations",url:"group__mpi__types__ops.html",children:[
{text:"mpi_type",url:"structmpi_1_1mpi__type.html",children:[
{text:"mpi_type<bool>",url:"structmpi_1_1mpi__type_3_01bool_01_4.html"},
{text:"mpi_type<char>",url:"structmpi_1_1mpi__type_3_01char_01_4.html"},
{text:"mpi_type<int>",url:"structmpi_1_1mpi__type_3_01int_01_4.html"},
{text:"mpi_type<long>",url:"structmpi_1_1mpi__type_3_01long_01_4.html"},
{text:"mpi_type<long long>",url:"structmpi_1_1mpi__type_3_01long_01long_01_4.html"},
{text:"mpi_type<double>",url:"structmpi_1_1mpi__type_3_01double_01_4.html"},
{text:"mpi_type<float>",url:"structmpi_1_1mpi__type_3_01float_01_4.html"},
{text:"mpi_type<std::complex<double>>",url:"structmpi_1_1mpi__type_3_01std_1_1complex_3_01double_01_4_01_4.html"},
{text:"mpi_type<unsigned int>",url:"structmpi_1_1mpi__type_3_01unsigned_01int_01_4.html"},
{text:"mpi_type<unsigned long>",url:"structmpi_1_1mpi__type_3_01unsigned_01long_01_4.html"},
{text:"mpi_type<unsigned long long>",url:"structmpi_1_1mpi__type_3_01unsigned_01long_01long_01_4.html"},
{text:"mpi_type<std::tuple>",url:"structmpi_1_1mpi__type.html"}]},
{text:"mpi_type_from_tie",url:"structmpi_1_1mpi__type__from__tie.html"},
{text:"get_mpi_type",url:"group__mpi__types__ops.html#ga03e748f7909f1e38c43019570c78657d"},
{text:"has_mpi_type",url:"group__mpi__types__ops.html#gac117479a485f170ca920a64ba1e4ac34"},
{text:"map_C_function",url:"group__mpi__types__ops.html#gae8618c4b71923a982ab66f1bf62549c9"},
{text:"map_add",url:"group__mpi__types__ops.html#gab81d56ee147c1034fea1bf001c1856e0"}]},
{text:"Collective MPI communication",url:"group__coll__comm.html",children:[
{text:"all_gather",url:"group__coll__comm.html#gafae74e49ad6ee44f66cf1211ba8cf54e"},
{text:"all_reduce",url:"group__coll__comm.html#gabda7358ee96ff22cfdc37b73630406ca"},
{text:"all_reduce_in_place",url:"group__coll__comm.html#ga85d54c696e3628d4c4cb00144b21af77"},
{text:"broadcast",url:"group__coll__comm.html#ga26c2a2de93ecfb78c235edada3b5f57b",children:[
{text:"mpi_broadcast",url:"group__coll__comm.html#ga7b441294b27fb668e3876294ba7fdc03"},
{text:"mpi_broadcast for std::pair",url:"group__coll__comm.html#gab331d0f1361f69228665b60f3262588a"},
{text:"mpi_broadcast for std::string",url:"group__coll__comm.html#gabcb8e2f22f68900179f4edd9ea175eab"},
{text:"mpi_broadcast for std::vector",url:"group__coll__comm.html#ga452e85429efcccd36141a49b2c7012cb"}]},
{text:"gather",url:"group__coll__comm.html#gaf697a18695e4a5344bbe76e67ae77277",children:[
{text:"mpi_gather for std::vector",url:"group__coll__comm.html#ga45186badcc4923ea5458a5c87b1cf01f"}]},
{text:"reduce",url:"group__coll__comm.html#ga8eaee05122bf70f15de0731b889e3949",children:[
{text:"mpi_reduce",url:"group__coll__comm.html#gab133dac60af76a9b16bc0d6983601a1f"},
{text:"mpi_reduce for std::pair",url:"group__coll__comm.html#ga596d7bf8804da7049839279e1670fcff"},
{text:"mpi_reduce for std::vector",url:"group__coll__comm.html#ga358e88080562398983adeb6bfd584477"}]},
{text:"reduce_in_place",url:"group__coll__comm.html#ga78f3e1fa0d66c5e4d4bfa1b12a75365f",children:[
{text:"mpi_reduce_in_place",url:"group__coll__comm.html#ga3f37de03067953c6038ce322e81df7e6"},
{text:"mpi_reduce_in_place for std::vector",url:"group__coll__comm.html#gae123b365235e205f4a4b992ff11b9722"}]},
{text:"scatter",url:"group__coll__comm.html#ga129f32374467c9b0f74a822a7e554588",children:[
{text:"mpi_scatter",url:"group__coll__comm.html#gad6813475336c8758f11e666476d57aab"}]}]},
{text:"Lazy MPI communication",url:"group__mpi__lazy.html",children:[
{text:"lazy",url:"structmpi_1_1lazy.html"},
{text:"gather tag",url:"structmpi_1_1tag_1_1gather.html"},
{text:"reduce tag",url:"structmpi_1_1tag_1_1reduce.html"},
{text:"scatter tag",url:"structmpi_1_1tag_1_1scatter.html"},
{text:"is_mpi_lazy",url:"group__mpi__lazy.html#gaea1d1d296f80ece0880a0d39aa8ca6bb"}]},
{text:"Event handling",url:"group__event__handling.html",children:[
{text:"monitor",url:"classmpi_1_1monitor.html"}]},
{text:"Utilities",url:"group__utilities.html",children:[
{text:"regular_t",url:"group__utilities.html#gabc2abaca95fbe7d5faf32c6998f20da8"},
{text:"chunk",url:"group__utilities.html#gae690fbc2d13c3ef957dad296412f4df4"},
{text:"chunk_length",url:"group__utilities.html#ga72cff436e2418ebdf5c905f1bb489d95"}]},
{text:"File List",url:"files.html"}]},
{text:"Changelog",url:"changelog.html"},
{text:"Issues",url:"issues.html"}]}
