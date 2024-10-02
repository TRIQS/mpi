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

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

TEST(MPI, AllEqual) {
  // check if a value is equal on all ranks
  mpi::communicator world;

  int val_i = 10;
  EXPECT_TRUE(mpi::all_equal(val_i, world));
  double val_d = 3.1415;
  EXPECT_TRUE(mpi::all_equal(val_d, world));
  std::vector<int> val_v = {1, 2, 3};
  EXPECT_TRUE(mpi::all_equal(val_v, world));

  if (world.size() > 1) {
    if (world.rank() == 1) val_i -= 1;
    EXPECT_FALSE(mpi::all_equal(val_i, world));
    if (world.rank() == 1) val_d -= 1.0;
    EXPECT_FALSE(mpi::all_equal(val_d, world));
    if (world.rank() == 1) val_v[0] -= 1;
    EXPECT_FALSE(mpi::all_equal(val_v, world));
  }
}

MPI_TEST_MAIN;
