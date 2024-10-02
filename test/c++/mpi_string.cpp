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

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <string>

TEST(MPI, StringBroadcast) {
  // broadcast a string
  mpi::communicator world;

  std::string s;
  if (world.rank() == 0) s = "Hello World";

  mpi::broadcast(s);

  EXPECT_EQ(s, std::string{"Hello World"});
}

TEST(MPI, StringGather) {
  // gather a string
  mpi::communicator world;
  std::string s{}, exp_s{};
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < i + 1; ++j) exp_s += "a";
    exp_s += std::to_string(i);
  }
  for (int i = 0; i < world.rank() + 1; ++i) s += "a";
  s += std::to_string(world.rank());

  // gather only on root
  auto s_gathered = mpi::gather(s);
  if (world.rank() == 0) EXPECT_EQ(s_gathered, exp_s);
  else EXPECT_TRUE(s_gathered.empty());

  // gather on all processes
  auto s_gathered_all = mpi::all_gather(s);
  EXPECT_EQ(s_gathered_all, exp_s);
}

MPI_TEST_MAIN;
