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

#include <mpi/vector.hpp>
#include <gtest/gtest.h>

#include <complex>

using namespace itertools;

TEST(MPI, vector_reduce) {

  mpi::communicator world;

  const int N = 7;
  using VEC   = std::vector<std::complex<double>>;

  VEC A(N), B;

  for (int i = 0; i < N; ++i) A[i] = i; //+ world.rank();

  B = mpi::all_reduce(A, world);

  VEC res(N);
  for (int i = 0; i < N; ++i) res[i] = world.size() * i; // +  world.size()*(world.size() - 1)/2;

  EXPECT_EQ(B, res);
}

// -----------------------------------

TEST(MPI, vector_gather_scatter) {

  mpi::communicator world;

  std::vector<std::complex<double>> A(7), B(7), AA(7);

  for (auto [i, v_i] : enumerate(A)) v_i = i + 1;

  B      = mpi::scatter(A, world);
  auto C = mpi::scatter(A, world);

  for (auto &x : B) x *= -1;
  for (auto &x : AA) x = 0;
  for (auto &x : A) x *= -1;

  AA = mpi::all_gather(B, world);

  EXPECT_EQ(A, AA);
}

MPI_TEST_MAIN;
