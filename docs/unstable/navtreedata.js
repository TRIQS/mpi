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
var NAVTREE =
[
  [ "TRIQS/mpi", "index.html", [
    [ "Overview", "index.html", "index" ],
    [ "Installation", "installation.html", [
      [ "Dependencies", "installation.html#dependencies", null ],
      [ "Installation steps", "installation.html#install_steps", null ],
      [ "Versions", "installation.html#versions", null ],
      [ "Custom CMake options", "installation.html#cmake_options", null ]
    ] ],
    [ "Integration in C++ projects", "integration.html", [
      [ "CMake", "integration.html#cmake", [
        [ "FetchContent", "integration.html#fetch", null ],
        [ "find_package", "integration.html#find_package", null ],
        [ "add_subdirectory", "integration.html#add_sub", null ]
      ] ],
      [ "Other", "integration.html#other", null ]
    ] ],
    [ "Examples", "examples.html", [
      [ "Compiling the examples", "examples.html#compiling", null ],
      [ "Example 1: Hello world!", "ex1.html", null ],
      [ "Example 2: Use monitor to communicate errors", "ex2.html", null ],
      [ "Example 3: Custom type and operator", "ex3.html", null ]
    ] ],
    [ "API Documentation", "documentation.html", [
      [ "MPI essentials", "group__mpi__essentials.html", [
        [ "communicator", "classmpi_1_1communicator.html", null ],
        [ "environment", "structmpi_1_1environment.html", null ],
        [ "is_initialized", "group__mpi__essentials.html#gaee54f343fdd8f1712ae521bd8ee69dfc", null ],
        [ "has_env", "group__mpi__essentials.html#ga590f450f6987d3e6c0398048515856b1", null ]
      ] ],
      [ "MPI datatypes and operations", "group__mpi__types__ops.html", [
        [ "mpi_type", "structmpi_1_1mpi__type.html", [
          [ "mpi_type<bool>", "structmpi_1_1mpi__type_3_01bool_01_4.html", null ],
          [ "mpi_type<char>", "structmpi_1_1mpi__type_3_01char_01_4.html", null ],
          [ "mpi_type<int>", "structmpi_1_1mpi__type_3_01int_01_4.html", null ],
          [ "mpi_type<long>", "structmpi_1_1mpi__type_3_01long_01_4.html", null ],
          [ "mpi_type<long long>", "structmpi_1_1mpi__type_3_01long_01long_01_4.html", null ],
          [ "mpi_type<double>", "structmpi_1_1mpi__type_3_01double_01_4.html", null ],
          [ "mpi_type<float>", "structmpi_1_1mpi__type_3_01float_01_4.html", null ],
          [ "mpi_type<std::complex<double>>", "structmpi_1_1mpi__type_3_01std_1_1complex_3_01double_01_4_01_4.html", null ],
          [ "mpi_type<unsigned int>", "structmpi_1_1mpi__type_3_01unsigned_01int_01_4.html", null ],
          [ "mpi_type<unsigned long>", "structmpi_1_1mpi__type_3_01unsigned_01long_01_4.html", null ],
          [ "mpi_type<unsigned long long>", "structmpi_1_1mpi__type_3_01unsigned_01long_01long_01_4.html", null ],
          [ "mpi_type<std::tuple>", "structmpi_1_1mpi__type.html", null ]
        ] ],
        [ "mpi_type_from_tie", "structmpi_1_1mpi__type__from__tie.html", null ],
        [ "get_mpi_type", "group__mpi__types__ops.html#ga03e748f7909f1e38c43019570c78657d", null ],
        [ "has_mpi_type", "group__mpi__types__ops.html#gac117479a485f170ca920a64ba1e4ac34", null ],
        [ "map_C_function", "group__mpi__types__ops.html#gae8618c4b71923a982ab66f1bf62549c9", null ],
        [ "map_add", "group__mpi__types__ops.html#gab81d56ee147c1034fea1bf001c1856e0", null ]
      ] ],
      [ "Collective MPI communication", "group__coll__comm.html", [
        [ "all_gather", "group__coll__comm.html#gafae74e49ad6ee44f66cf1211ba8cf54e", null ],
        [ "all_reduce", "group__coll__comm.html#gabda7358ee96ff22cfdc37b73630406ca", null ],
        [ "all_reduce_in_place", "group__coll__comm.html#ga85d54c696e3628d4c4cb00144b21af77", null ],
        [ "broadcast", "group__coll__comm.html#ga26c2a2de93ecfb78c235edada3b5f57b", [
          [ "mpi_broadcast", "group__coll__comm.html#ga7b441294b27fb668e3876294ba7fdc03", null ],
          [ "mpi_broadcast for std::pair", "group__coll__comm.html#gab331d0f1361f69228665b60f3262588a", null ],
          [ "mpi_broadcast for std::string", "group__coll__comm.html#gabcb8e2f22f68900179f4edd9ea175eab", null ],
          [ "mpi_broadcast for std::vector", "group__coll__comm.html#ga452e85429efcccd36141a49b2c7012cb", null ]
        ] ],
        [ "gather", "group__coll__comm.html#gaf697a18695e4a5344bbe76e67ae77277", [
          [ "mpi_gather for std::vector", "group__coll__comm.html#ga45186badcc4923ea5458a5c87b1cf01f", null ]
        ] ],
        [ "reduce", "group__coll__comm.html#ga8eaee05122bf70f15de0731b889e3949", [
          [ "mpi_reduce", "group__coll__comm.html#gab133dac60af76a9b16bc0d6983601a1f", null ],
          [ "mpi_reduce for std::pair", "group__coll__comm.html#ga596d7bf8804da7049839279e1670fcff", null ],
          [ "mpi_reduce for std::vector", "group__coll__comm.html#ga358e88080562398983adeb6bfd584477", null ]
        ] ],
        [ "reduce_in_place", "group__coll__comm.html#ga78f3e1fa0d66c5e4d4bfa1b12a75365f", [
          [ "mpi_reduce_in_place", "group__coll__comm.html#ga3f37de03067953c6038ce322e81df7e6", null ],
          [ "mpi_reduce_in_place for std::vector", "group__coll__comm.html#gae123b365235e205f4a4b992ff11b9722", null ]
        ] ],
        [ "scatter", "group__coll__comm.html#ga129f32374467c9b0f74a822a7e554588", [
          [ "mpi_scatter", "group__coll__comm.html#gad6813475336c8758f11e666476d57aab", null ]
        ] ]
      ] ],
      [ "Lazy MPI communication", "group__mpi__lazy.html", [
        [ "lazy", "structmpi_1_1lazy.html", null ],
        [ "gather tag", "structmpi_1_1tag_1_1gather.html", null ],
        [ "reduce tag", "structmpi_1_1tag_1_1reduce.html", null ],
        [ "scatter tag", "structmpi_1_1tag_1_1scatter.html", null ],
        [ "is_mpi_lazy", "group__mpi__lazy.html#gac297d27e56328194184d9474f8bb5f87", null ]
      ] ],
      [ "Error handling", "group__err__handling.html", [
        [ "monitor", "classmpi_1_1monitor.html", null ]
      ] ],
      [ "Utilities", "group__utilities.html", [
        [ "regular_t", "group__utilities.html#gabc2abaca95fbe7d5faf32c6998f20da8", null ],
        [ "chunk", "group__utilities.html#gae690fbc2d13c3ef957dad296412f4df4", null ],
        [ "chunk_length", "group__utilities.html#ga72cff436e2418ebdf5c905f1bb489d95", null ]
      ] ],
      [ "File List", "files.html", "files" ]
    ] ],
    [ "Changelog", "changelog.html", null ],
    [ "Issues", "issues.html", null ]
  ] ]
];

var NAVTREEINDEX =
[
"changelog.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';