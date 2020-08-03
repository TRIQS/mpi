// Copyright (c) 2020 Simons Foundation
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

#include <mpi/mpi.hpp>
#include <gtest/gtest.h>

using namespace mpi;

TEST(Comm, split) {

  communicator world;
  int rank = world.rank();

  ASSERT_TRUE(2 == world.size() or 4 == world.size());

  int colors[] = {0, 2, 1, 1};
  int keys[]   = {5, 7, 13, 18};

  auto comm = world.split(colors[rank], keys[rank]);

  int comm_sizes[] = {1, 1, 2, 2};
  int comm_ranks[] = {0, 0, 0, 1};

  EXPECT_EQ(comm_sizes[rank], comm.size());
  EXPECT_EQ(comm_ranks[rank], comm.rank());
}

MPI_TEST_MAIN;
