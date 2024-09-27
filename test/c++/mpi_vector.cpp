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

#include "./non_mpi_t.hpp"

#include <gtest/gtest.h>
#include <itertools/itertools.hpp>
#include <mpi/pair.hpp>
#include <mpi/string.hpp>
#include <mpi/vector.hpp>

#include <complex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

TEST(MPI, VectorBroadcastMPIType) {
  // broadcast a vector with an MPI type
  mpi::communicator world;
  std::vector<int> vec(5, 0);
  if (world.rank() == 0) {
    std::iota(vec.begin(), vec.end(), 0);
  } else {
    vec.clear();
  }
  mpi::broadcast(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i], i);
}

TEST(MPI, VectorBroadcastTypeWithSpezializedMPIBroadcast) {
  // broadcast a vector with a type that has a specialized mpi_broadcast
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5);
  if (world.rank() == 0) {
    for (int i = 0; i < 5; ++i) vec[i].a = i;
  }
  mpi::broadcast(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i);
}

TEST(MPI, VectorReduceInPlaceMPIType) {
  // in-place reduce a vector with an MPI type
  mpi::communicator world;
  std::vector<int> vec{0, 1, 2, 3, 4};
  mpi::reduce_in_place(vec, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i], i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i], i);

  // in-place allreduce a vector with an MPI type
  std::iota(vec.begin(), vec.end(), 0);
  mpi::all_reduce_in_place(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i], i * world.size());
}

TEST(MPI, VectorReduceInPlaceTypeWithSpezializedMPIReduceInPlace) {
  // in-place reduce a vector with a type that has a specialized mpi_reduce_in_place
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5);
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::reduce_in_place(vec, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i * world.size());
  else
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i);

  // in-place allreduce a vector with a type that has a specialized mpi_reduce_in_place
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  mpi::all_reduce_in_place(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec[i].a, i * world.size());
}

TEST(MPI, VectorReduceMPIType) {
  // reduce a vector with complex numbers
  mpi::communicator world;
  using vec_type = std::vector<std::complex<double>>;
  const int size = 7;
  vec_type vec(size);
  for (int i = 0; i < size; ++i) vec[i] = std::complex<double>(i, -i);
  auto vec_reduced = mpi::reduce(vec, world);
  if (world.rank() == 0)
    for (int i = 0; i < size; ++i) EXPECT_EQ(vec_reduced[i], std::complex<double>(i * world.size(), -i * world.size()));
  else
    EXPECT_TRUE(vec_reduced.empty());

  // allreduce a vector with complex numbers
  vec_reduced = mpi::all_reduce(vec, world);
  for (int i = 0; i < size; ++i) EXPECT_EQ(vec_reduced[i], std::complex<double>(i * world.size(), -i * world.size()));
}

TEST(MPI, VectorReduceTypeWithSpezializedMPIReduce) {
  // reduce a vector with a type that has a specialized mpi_reduce
  mpi::communicator world;
  std::vector<non_mpi_t> vec(5);
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  auto vec_reduced = mpi::reduce(vec, world);
  if (world.rank() == 0)
    for (int i = 0; i < 5; ++i) EXPECT_EQ(vec_reduced[i].a, i * world.size());
  else
    EXPECT_TRUE(vec_reduced.empty());

  // allreduce a vector with a type that has a specialized mpi_reduce
  for (int i = 0; i < 5; ++i) vec[i].a = i;
  auto vec_reduced_all = mpi::all_reduce(vec, world);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(vec_reduced_all[i].a, i * world.size());
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

TEST(MPI, VectorScatterSizeZero) {
  // pass a vector of size 0 to scatter
  mpi::communicator world;
  std::vector<int> v = {1, 2, 3};
  if (world.rank() == 0) v.clear();
  auto res = mpi::scatter(v, world);
  EXPECT_TRUE(res.empty());
}

MPI_TEST_MAIN;
