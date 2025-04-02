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
#include <vector>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test gathering a range of objects.
template <typename T> void test_gather_range(std::vector<T> const &values, std::vector<T> const &result) {
  mpi::communicator world;

  // gather on different roots
  for (int root = 0; root < world.size(); ++root) {
    // gather spans into a view of a vector
    std::vector<T> vec(result.size() * 2, T{0});
    mpi::gather_range(std::span{values}, std::ranges::drop_view(vec, result.size()), world, root);
    if (world.rank() == root) {
      expect_range_eq(std::ranges::drop_view(vec, result.size()), result);
      expect_range_eq(std::ranges::take_view(vec, result.size()), std::vector<T>(result.size(), T{0}));
    } else {
      expect_range_eq(vec, std::vector<T>(result.size() * 2, T{0}));
    }
  }

  // allgather vectors into an oversized vector
  std::vector<T> vec(result.size() * 2, T{0});
  mpi::gather_range(values, std::span{vec.begin(), result.size()}, world, 0, true);
  expect_range_eq(std::ranges::take_view(vec, result.size()), result);
  expect_range_eq(std::ranges::drop_view(vec, result.size()), std::vector<T>(result.size(), T{0}));
}

TEST(MPI, GatherIntegerRange) {
  mpi::communicator world;
  std::vector<int> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i);
  };
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank());
  test_gather_range(values, result);
}

TEST(MPI, GatherComplexRange) {
  mpi::communicator world;
  std::vector<std::complex<double>> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i, -i);
  }
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank(), -world.rank());
  test_gather_range(values, result);
}

TEST(MPI, GatherCustomMPITypeRange) {
  mpi::communicator world;
  std::vector<mpi_t> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i);
  }
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank());
  test_gather_range(values, result);
}

MPI_TEST_MAIN;
