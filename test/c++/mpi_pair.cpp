// Copyright (c) 2021-2024 Simons Foundation
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
#include <mpi/pair.hpp>
#include <mpi/string.hpp>

#include <complex>
#include <string>
#include <utility>

TEST(MPI, PairBroadcast) {
  // broadcast a pair consisting of a string and a complex number
  std::pair<std::string, std::complex<double>> p;

  auto str  = std::string{"Hello"};
  auto cplx = std::complex<double>(1.0, 2.0);

  mpi::communicator world;
  if (world.rank() == 0) p = {str, cplx};

  mpi::broadcast(p);
  auto [str_bc, cplx_bc] = p;
  EXPECT_EQ(str, str_bc);
  EXPECT_EQ(cplx, cplx_bc);
}

TEST(MPI, PairReduce) {
  // reduce a pair of integers
  mpi::communicator world;
  auto r = world.rank();
  auto p = std::pair{1, r};

  auto [r1, r2] = mpi::all_reduce(p);
  auto nr       = world.size();
  EXPECT_EQ(r1, nr);
  EXPECT_EQ(r2, nr * (nr - 1) / 2);
}

MPI_TEST_MAIN;
