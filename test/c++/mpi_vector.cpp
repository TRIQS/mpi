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

#include "./custom_types.hpp"

#include <gtest/gtest.h>
#include <itertools/itertools.hpp>
#include <mpi/mpi.hpp>

#include <complex>
#include <string>
#include <utility>
#include <vector>

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

TEST(MPI, VectorGatherPair) {
  // gather a vector of pairs
  mpi::communicator world;
  auto const rank          = world.rank();
  auto const gathered_size = (world.size() + 1) * world.size() / 2;
  std::vector<std::pair<int, std::string>> vec(world.rank() + 1);
  for (int i = 0; i < vec.size(); ++i) {
    vec[i].first  = i + rank * (rank + 1) / 2;
    vec[i].second = std::to_string(vec[i].first);
  }
  auto vec_gathered = mpi::all_gather(vec, world);
  for (int i = 0; i < gathered_size; ++i) EXPECT_EQ(vec_gathered[i], std::make_pair(i, std::to_string(i)));
}

TEST(MPI, VectorGatherOnlyOnRoot) {
  // gather a vector only on root
  mpi::communicator world;
  std::vector<int> v = {1, 2, 3};
  auto res           = mpi::gather(v, world);
  if (world.rank() == 0) {
    auto exp_res = v;
    for (int i = 1; i < world.size(); ++i) exp_res.insert(exp_res.end(), v.begin(), v.end());
    EXPECT_EQ(res, exp_res);
  } else {
    EXPECT_TRUE(res.empty());
  }
}

MPI_TEST_MAIN;
