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

#include <array>
#include <complex>
#include <ranges>
#include <string>
#include <utility>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test broadcasting arrays.
template <typename T> void test_broadcast_array(std::array<T, 5> const &root_values) {
  mpi::communicator world;
  auto arr = root_values;

  // broadcast an array from different roots
  for (int root = 0; root < world.size(); ++root) {
    arr = {};
    if (world.rank() == root) arr = root_values;
    mpi::broadcast(arr, world, root);
    expect_range_eq(arr, root_values);
  }

  // broadcast an empty array
  std::array<T, 0> empty_arr{};
  mpi::broadcast(empty_arr, world);
  expect_range_eq(arr, root_values);
}

TEST(MPI, BroadcastIntegerArray) { test_broadcast_array(std::array<int, 5>{1, 2, 3, 4, 5}); }

TEST(MPI, BroadcastComplexArray) {
  using namespace std::complex_literals;
  test_broadcast_array(std::array<std::complex<double>, 5>{1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i, 5.0 - 5.0i});
}

TEST(MPI, BroadcastCustomMPITypeArray) { test_broadcast_array(std::array<mpi_t, 5>{mpi_t{1}, mpi_t{2}, mpi_t{3}, mpi_t{4}, mpi_t{5}}); }

TEST(MPI, BroadcastCustomNonMPITypeArray) {
  test_broadcast_array(std::array<non_mpi_t, 5>{non_mpi_t{1}, non_mpi_t{2}, non_mpi_t{3}, non_mpi_t{4}, non_mpi_t{5}});
}

TEST(MPI, BroadcastStringArray) { test_broadcast_array(std::array<std::string, 5>{"Hello", "World", "MPI", "Broadcast", "Array"}); }

TEST(MPI, BroadcastPairArray) {
  test_broadcast_array(std::array<std::pair<int, std::string>, 5>{{{1, "Hello"}, {2, "World"}, {3, "MPI"}, {4, "Broadcast"}, {5, "Array"}}});
}

TEST(MPI, BroadcastArrayOfDoubleArrays) {
  std::array<std::array<double, 2>, 5> root_values{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 2; ++j) root_values[i][j] = i * 2 + j;
  }
  test_broadcast_array(root_values);
}

MPI_TEST_MAIN;
