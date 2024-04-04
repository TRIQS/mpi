// Copyright (c) 2020-2021 Simons Foundation
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
// Authors: Nils Wentzell

#include <gtest/gtest.h>
#include <itertools/itertools.hpp>
#include <mpi/pair.hpp>
#include <mpi/string.hpp>
#include <mpi/vector.hpp>

#include <complex>
#include <string>
#include <utility>
#include <vector>

TEST(MPI, VectorReduce) {
  // reduce a vector of complex numbers
  mpi::communicator world;
  using vec_type = std::vector<std::complex<double>>;

  const int size = 7;
  vec_type vec(size), reduced_vec;

  for (int i = 0; i < size; ++i) vec[i] = i;

  reduced_vec = mpi::all_reduce(vec, world);

  vec_type exp_vec(size);
  for (int i = 0; i < size; ++i) exp_vec[i] = world.size() * i;

  EXPECT_EQ(reduced_vec, exp_vec);
}

TEST(MPI, EmptyVectorReduce) {
  // reduce an empty vector
  mpi::communicator world;
  std::vector<double> v1{};
  std::vector<double> v2 = mpi::reduce(v1, world);
}

TEST(MPI, VectorGatherScatter) {
  // scatter and gather a vector of complex numbers
  mpi::communicator world;

  std::vector<std::complex<double>> vec(7), scattered_vec(7), gathered_vec(7, {0.0, 0.0});

  for (auto [i, v_i] : itertools::enumerate(vec)) v_i = static_cast<double>(i) + 1.0;

  scattered_vec = mpi::scatter(vec, world);
  auto tmp      = mpi::scatter(vec, world);

  for (auto &x : scattered_vec) x *= -1;
  for (auto &x : vec) x *= -1;

  gathered_vec = mpi::all_gather(scattered_vec, world);

  EXPECT_EQ(vec, gathered_vec);
}

TEST(MPI, VectorGatherScatterPair) {
  // scatter and gather a vector of pairs
  auto v = std::vector<std::pair<int, std::string>>{{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}};

  auto vsct = mpi::scatter(v);
  auto vgth = mpi::all_gather(vsct);

  mpi::communicator world;
  if (world.size() > 1) { EXPECT_NE(vsct, vgth); }
  EXPECT_EQ(v, vgth);
}

MPI_TEST_MAIN;
