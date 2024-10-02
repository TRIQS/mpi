// Copyright (c) 2020-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Nils Wentzell

#include "./non_mpi_t.hpp"

#include <gtest/gtest.h>
#include <itertools/itertools.hpp>
#include <mpi/mpi.hpp>

#include <complex>
#include <numeric>
#include <tuple>

TEST(MPI, ArrayBroadcastMPIType) {
  // broadcast an array with an MPI type
  mpi::communicator world;
  std::array<int, 5> arr{};
  if (world.rank() == 0) std::iota(arr.begin(), arr.end(), 0);
  mpi::broadcast(arr, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i);
}

TEST(MPI, ArrayBroadcastTypeWithSpezializedMPIBroadcast) {
  // broadcast an array with a type that has a specialized mpi_broadcast
  mpi::communicator world;
  std::array<non_mpi_t, 5> arr{};
  if (world.rank() == 0) {
    for (int i = 0; i < 5; ++i) arr[i].a = i;
  }
  mpi::broadcast(arr, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i].a, i);
}

TEST(MPI, ArrayReduceInPlaceMPIType) {
  // in-place reduce an array with an MPI type
  mpi::communicator world;
  std::array<int, 5> arr{0, 1, 2, 3, 4};
  mpi::reduce_in_place(arr, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i);

  // in-place allreduce an array with an MPI type
  std::iota(arr.begin(), arr.end(), 0);
  mpi::all_reduce_in_place(arr, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i * world.size());
}

TEST(MPI, ArrayReduceInPlaceTypeWithSpezializedMPIReduceInPlace) {
  // in-place reduce an array with a type that has a specialized mpi_reduce_in_place
  mpi::communicator world;
  std::array<non_mpi_t, 5> arr{};
  for (int i = 0; i < 5; ++i) arr[i].a = i;
  mpi::reduce_in_place(arr, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i].a, i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i].a, i);

  // in-place allreduce an array with a type that has a specialized mpi_reduce_in_place
  for (int i = 0; i < 5; ++i) arr[i].a = i;
  mpi::all_reduce_in_place(arr, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i].a, i * world.size());
}

TEST(MPI, ArrayReduceMPIType) {
  // reduce an array with complex numbers
  mpi::communicator world;
  using arr_type = std::array<std::complex<double>, 7>;
  const int size = 7;
  arr_type arr{};
  for (int i = 0; i < size; ++i) arr[i] = std::complex<double>(i, -i);
  auto arr_reduced = mpi::reduce(arr, world);
  if (world.rank() == 0)
    for (int i = 0; i < size; ++i) EXPECT_EQ(arr_reduced[i], std::complex<double>(i * world.size(), -i * world.size()));
  else
    EXPECT_EQ(arr_reduced, arr_type{});

  // allreduce an array with complex numbers
  auto arr_reduced_all = mpi::all_reduce(arr, world);
  for (int i = 0; i < size; ++i) EXPECT_EQ(arr_reduced_all[i], std::complex<double>(i * world.size(), -i * world.size()));
}

TEST(MPI, EmptyArrayReduce) {
  // reduce an empty array
  mpi::communicator world;
  std::array<double, 0> arr{};
  std::ignore = mpi::reduce(arr, world);
}

MPI_TEST_MAIN;
