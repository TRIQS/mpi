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

#include <vector>

TEST(MPI, GroupDefaultConstructor) {
  mpi::group g;
  EXPECT_TRUE(g.is_null());
}

TEST(MPI, GroupCommWorld) {
  mpi::communicator world;

  mpi::group g(world);
  EXPECT_EQ(g.rank(), world.rank());
  EXPECT_EQ(g.size(), world.size());

  // move operations
  auto g2 = std::move(g);
  EXPECT_EQ(g2.rank(), world.rank());
  EXPECT_EQ(g2.size(), world.size());
  EXPECT_TRUE(g.is_null());
}

TEST(MPI, GroupInclude) {
  mpi::communicator world;
  mpi::group g(world);

  // include every second rank
  std::vector<int> ranks;
  for (int i = 0; i < world.size(); i += 2) ranks.push_back(i);
  auto g2 = g.include(ranks);
  EXPECT_EQ(g2.size(), ranks.size());
  if (std::ranges::find(ranks, world.rank()) != ranks.end()) {
    EXPECT_EQ(world.rank(), ranks[g2.rank()]);
  } else {
    EXPECT_EQ(g2.rank(), MPI_UNDEFINED);
  }

  // include every second rank (starting from the back)
  ranks.clear();
  for (int i = world.size() - 1; i >= 0; i -= 2) ranks.push_back(i);
  auto g3 = g.include(ranks);
  EXPECT_EQ(g3.size(), ranks.size());
  if (std::ranges::find(ranks, world.rank()) != ranks.end()) {
    EXPECT_EQ(world.rank(), ranks[g3.rank()]);
  } else {
    EXPECT_EQ(g3.rank(), MPI_UNDEFINED);
  }
}

MPI_TEST_MAIN;
