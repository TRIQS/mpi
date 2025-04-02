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

#include "./custom_types.hpp"

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <numeric>
#include <vector>

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
