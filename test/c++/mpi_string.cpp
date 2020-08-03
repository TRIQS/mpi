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

#include <mpi/string.hpp>
#include <gtest/gtest.h>

TEST(MPI, string_broadcast) {

  mpi::communicator world;

  std::string s;
  if (world.rank() == 0) s = "Hello World";

  mpi::broadcast(s);

  EXPECT_EQ(s, std::string{"Hello World"});
}

MPI_TEST_MAIN;
