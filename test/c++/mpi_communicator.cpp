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

TEST(MPI, CommunicatorDuplicateWorld) {
  mpi::communicator world;

  // skit the rest of test if there is no active MPI runtime
  if (!mpi::has_env) return;

  // duplicate and check the communicator
  auto dup = world.duplicate();
  EXPECT_EQ(world.rank(), dup.rank());
  EXPECT_EQ(world.size(), dup.size());
  EXPECT_EQ(MPI_COMM_WORLD, world.get());
  EXPECT_NE(world.get(), dup.get());

  // free the communicator
  EXPECT_FALSE(dup.is_null());
  dup.free();
  EXPECT_TRUE(dup.is_null());
}

TEST(MPI, CommunicatorSplitAndDuplicate) {
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

  // duplicate the split communicator and check
  auto dup = comm.duplicate();
  EXPECT_EQ(comm.rank(), dup.rank());
  EXPECT_EQ(comm.size(), dup.size());

  // free the communicators
  EXPECT_FALSE(dup.is_null());
  EXPECT_FALSE(comm.is_null());
  dup.free();
  comm.free();
  EXPECT_TRUE(dup.is_null());
  EXPECT_TRUE(comm.is_null());
}

TEST(MPI_Window, CommunicatorSplitShared) {
  mpi::communicator world;
  [[maybe_unused]] auto shm = world.split_shared();
}

TEST(MPI, SharedCommunicatorDefaultConstructor) {
  mpi::shared_communicator comm{};
  EXPECT_TRUE(comm.is_null());
}

MPI_TEST_MAIN;
