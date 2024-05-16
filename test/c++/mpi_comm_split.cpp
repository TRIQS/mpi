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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <array>

TEST(MPI, CommunicatorSplit) {
  mpi::communicator world;
  int rank = world.rank();

  // skip test if only one rank in communicator
  if (world.size() == 1) return;

  // only works for 2 or 4 processes
  ASSERT_TRUE(2 == world.size() or 4 == world.size());

  // split the communicator into 2 (3) for 2 (4) processes
  auto colors = std::array{0, 2, 1, 1};
  auto keys   = std::array{5, 7, 13, 18};
  auto comm   = world.split(colors[rank], keys[rank]);

  // check results
  auto exp_sizes = std::array{1, 1, 2, 2};
  auto exp_ranks = std::array{0, 0, 0, 1};
  EXPECT_EQ(exp_sizes[rank], comm.size());
  EXPECT_EQ(exp_ranks[rank], comm.rank());
}

MPI_TEST_MAIN;
