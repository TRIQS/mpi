// Copyright (c) 2022-2024 Simons Foundation
// Copyright (c) 2022 Hugo U.R. Strand
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
// Authors: Thomas Hahn, Hugo U.R. Strand

#include "./non_mpi_t.hpp"

#include <gtest/gtest.h>
#include <mpi/ranges.hpp>
#include <mpi/vector.hpp>

#include <array>
#include <numeric>
#include <vector>

TEST(MPI, RangesBroadcastMPIType) {
  // broadcast a range with an MPI type
  mpi::communicator world;
  std::array<int, 5> arr{};
  if (world.rank() == 0) {
    for (int i = 0; i < 5; ++i) arr[i] = i;
  }
  mpi::broadcast_range(arr, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i);
}

TEST(MPI, RangesBroadcastTypeWithSpezializedMPIBroadcast) {
  // broadcast a range with a type that has a specialized mpi_broadcast
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5);
  if (world.rank() == 0) {
    for (int i = 0; i < 5; ++i) vec[i].a = i;
  }
  mpi::broadcast_range(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i);
}

TEST(MPI, RangesReduceInPlaceMPIType) {
  // in-place reduce a range with an MPI type
  mpi::communicator world;
  std::array<int, 5> arr{0, 1, 2, 3, 4};
  mpi::reduce_in_place_range(arr, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i);

  // in-place allreduce a range with an MPI type
  arr = {0, 1, 2, 3, 4};
  mpi::reduce_in_place_range(arr, world, 0, true);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr[i], i * world.size());
}

TEST(MPI, RangesReduceInPlaceTypeWithSpezializedMPIReduceInPlace) {
  // in-place reduce a range with a type that has a specialized mpi_reduce_in_place
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5);
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::reduce_in_place_range(vec, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i);

  // in-place allreduce a range with a type that has a specialized mpi_reduce_in_place
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::reduce_in_place_range(vec, world, 0, true);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i * world.size());
}

TEST(MPI, RangesReduceMPIType) {
  // reduce a range with an MPI type
  mpi::communicator world;
  std::array<int, 5> arr{0, 1, 2, 3, 4}, arr_red{};
  mpi::reduce_range(arr, arr_red, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr_red[i], i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(arr_red[i], 0);

  // allreduce a range with an MPI type
  arr     = {0, 1, 2, 3, 4};
  arr_red = {};
  mpi::reduce_range(arr, arr_red, world, 0, true);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(arr_red[i], i * world.size());
}

TEST(MPI, RangesReduceTypeWithSpezializedMPIReduceInPlace) {
  // reduce a range with a type that has a specialized mpi_reduce_in_place
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5, non_mpi_t{}), vec_red(5, non_mpi_t{});
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::reduce_range(vec, vec_red, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec_red[i].a, i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec_red[i].a, non_mpi_t{}.a);

  // allreduce a range with a type that has a specialized mpi_reduce_in_place
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::reduce_range(vec, vec_red, world, 0, true);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec_red[i].a, i * world.size());
}

TEST(MPI, RangesScatterMPIType) {
  // scatter a range with an MPI type
  mpi::communicator world;
  auto const rank = world.rank();
  auto sizes      = std::vector<int>(world.size());
  for (int i = 0; i < world.size(); ++i) sizes[i] = static_cast<int>(mpi::chunk_length(10, world.size(), i));
  auto acc_sizes = std::vector<int>(world.size() + 1, 0);
  std::partial_sum(sizes.begin(), sizes.end(), std::next(acc_sizes.begin()));
  std::vector<int> vec(10, 0), vec_scattered(sizes[rank], 0);
  if (rank == 0) {
    for (int i = 0; i < 10; ++i) vec[i] = i;
  }
  mpi::scatter_range(vec, vec_scattered, 10, world, 0);
  for (int i = 0; i < sizes[rank]; ++i) EXPECT_EQ(vec_scattered[i], i + acc_sizes[rank]);
}

TEST(MPI, RangesGatherMPIType) {
  // gather a range with an MPI type
  mpi::communicator world;
  auto const rank          = world.rank();
  auto const gathered_size = (world.size() + 1) * world.size() / 2;
  std::vector<int> vec(world.rank() + 1, 0), vec_gathered(gathered_size, 0);
  std::iota(vec.begin(), vec.end(), rank * (rank + 1) / 2);
  mpi::gather_range(vec, vec_gathered, gathered_size, world, 0, false);
  if (rank == 0) {
    for (int i = 0; i < gathered_size; ++i) EXPECT_EQ(vec_gathered[i], i);
  }
}

TEST(MPI, RangesGatherTypeWithSpecializedMPIBroadcast) {
  // gather a range with a type that has a specialized mpi_broadcast
  mpi::communicator world;
  auto const rank          = world.rank();
  auto const gathered_size = (world.size() + 1) * world.size() / 2;
  std::vector<non_mpi_t> vec(world.rank() + 1, non_mpi_t{}), vec_gathered(gathered_size, non_mpi_t{});
  for (int i = 0; i < vec.size(); ++i) vec[i].a = i + rank * (rank + 1) / 2;

  // providing the size of the output range
  mpi::gather_range(vec, vec_gathered, gathered_size, world, 0, true);
  for (int i = 0; i < gathered_size; ++i) EXPECT_EQ(vec_gathered[i].a, i);
}

MPI_TEST_MAIN;
