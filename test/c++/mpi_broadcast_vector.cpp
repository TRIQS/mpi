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

#include "./custom_types.hpp"

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <complex>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test broadcasting vectors.
template <typename T> void test_broadcast_vector(std::vector<T> const &root_values) {
  mpi::communicator world;
  auto vec = root_values;

  // broadcast a vector from different roots
  for (int root = 0; root < world.size(); ++root) {
    vec.clear();
    if (world.rank() == root) vec = root_values;
    mpi::broadcast(vec, world, root);
    expect_range_eq(vec, root_values);
  }

  // broadcast an empty vector
  if (world.rank() == 0) {
    vec.clear();
    mpi::broadcast(vec, world);
    EXPECT_TRUE(vec.empty());
  } else {
    vec = root_values;
    mpi::broadcast(vec, world);
    EXPECT_TRUE(vec.empty());
  }
}

TEST(MPI, BroadcastIntegerVector) { test_broadcast_vector(std::vector<int>{1, 2, 3, 4, 5}); }

TEST(MPI, BroadcastComplexVector) {
  using namespace std::complex_literals;
  test_broadcast_vector(std::vector<std::complex<double>>{1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i, 5.0 - 5.0i});
}

TEST(MPI, BroadcastCustomMPITypeVector) { test_broadcast_vector(std::vector<mpi_t>{mpi_t{1}, mpi_t{2}, mpi_t{3}, mpi_t{4}, mpi_t{5}}); }

TEST(MPI, BroadcastCustomNonMPITypeVector) {
  test_broadcast_vector(std::vector<non_mpi_t>{non_mpi_t{1}, non_mpi_t{2}, non_mpi_t{3}, non_mpi_t{4}, non_mpi_t{5}});
}

TEST(MPI, BroadcastStringVector) { test_broadcast_vector(std::vector<std::string>{"Hello", "World", "MPI", "Broadcast", "Array"}); }

TEST(MPI, BroadcastPairVector) {
  test_broadcast_vector(std::vector<std::pair<int, std::string>>{{{1, "Hello"}, {2, "World"}, {3, "MPI"}, {4, "Broadcast"}, {5, "Array"}}});
}

TEST(MPI, BroadcastVectorOfDoubleVectors) {
  std::vector<std::vector<double>> root_values(5, std::vector<double>(2));
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 2; ++j) root_values[i][j] = i * 2 + j;
  }
  test_broadcast_vector(root_values);
}

MPI_TEST_MAIN;
