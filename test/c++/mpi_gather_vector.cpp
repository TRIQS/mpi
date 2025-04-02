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

// Test gathering vectors.
template <typename T> void test_gather_vector(std::vector<T> const &values, std::vector<T> const &result) {
  mpi::communicator world;

  // gather on different roots
  for (int root = 0; root < world.size(); ++root) {
    if constexpr (mpi::has_mpi_type<T>) {
      // gather vectors into a new vector
      auto vec = mpi::gather(values, world, root);
      if (world.rank() == root)
        expect_range_eq(vec, result);
      else
        EXPECT_TRUE(vec.empty());

      // gather vectors into an existing vector
      vec.clear();
      mpi::gather_into(values, vec, world, root);
      if (world.rank() == root)
        expect_range_eq(vec, result);
      else
        EXPECT_TRUE(vec.empty());
    }

    // gather empty vectors
    auto vec = mpi::gather(std::vector<T>{}, world, root);
    EXPECT_TRUE(vec.empty());
  }

  // allgather vectors into a new vector
  auto vec = mpi::all_gather(values, world);
  expect_range_eq(vec, result);

  // allgather vectors into an existing vector
  vec.clear();
  mpi::all_gather_into(values, vec, world);
  expect_range_eq(vec, result);
}

TEST(MPI, GatherIntegerVector) {
  mpi::communicator world;
  std::vector<int> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i);
  }
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank());
  test_gather_vector(values, result);
}

TEST(MPI, GatherComplexVector) {
  mpi::communicator world;
  std::vector<std::complex<double>> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i, -i);
  }
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank(), -world.rank());
  test_gather_vector(values, result);
}

TEST(MPI, GatherCustomMPITypeVector) {
  mpi::communicator world;
  std::vector<mpi_t> values, result;
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < 2 * (i + 1); ++j) result.emplace_back(i);
  }
  for (int i = 0; i < 2 * (world.rank() + 1); ++i) values.emplace_back(world.rank());
  test_gather_vector(values, result);
}

MPI_TEST_MAIN;
