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
#include <vector>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test gathering single values/objects.
template <typename T> void test_gather(std::vector<T> result) {
  mpi::communicator world;

  // gather from different roots
  for (int root = 0; root < world.size(); ++root) {
    // gather single objects into a vector
    auto vec = mpi::gather(result[world.rank()], world, root);
    if (world.rank() == root)
      expect_range_eq(vec, result);
    else
      EXPECT_TRUE(vec.empty());

    // gather single objects into an existing vector
    if (world.rank() == root) {
      vec.assign(world.size(), T{0});
      mpi::gather_into(result[world.rank()], vec, world, root);
      expect_range_eq(vec, result);
    } else {
      vec.clear();
      mpi::gather_into(result[world.rank()], vec, world, root);
      EXPECT_TRUE(vec.empty());
    }
  }

  // allgather single objects into a vector
  auto vec = mpi::all_gather(result[world.rank()], world);
  expect_range_eq(vec, result);

  // allgather single objects into an existing vector
  vec.assign(world.size(), T{0});
  mpi::all_gather_into(result[world.rank()], vec, world);
  expect_range_eq(vec, result);
}

TEST(MPI, GatherInteger) {
  mpi::communicator world;
  std::vector<int> result(world.size());
  for (int i = 0; i < world.size(); ++i) result[i] = i + 1;
  test_gather(result);
}

TEST(MPI, GatherComplex) {
  mpi::communicator world;
  std::vector<std::complex<double>> result(world.size());
  for (int i = 0; i < world.size(); ++i) result[i] = std::complex<double>{i + 1.0, -(i + 1.0)};
  test_gather(result);
}

TEST(MPI, GatherCustomMPIType) {
  mpi::communicator world;
  std::vector<mpi_t> result(world.size());
  for (int i = 0; i < world.size(); ++i) result[i] = mpi_t{i + 1};
  test_gather(result);
}

TEST(MPI, GatherCustomNonMPIType) {
  mpi::communicator world;
  std::vector<non_mpi_t> result(world.size());
  for (int i = 0; i < world.size(); ++i) result[i] = non_mpi_t{i + 1};
  test_gather(result);
}

// Test gathering a string.
TEST(MPI, GatherString) {
  mpi::communicator world;
  std::string str{}, result{};
  for (int i = 0; i < world.size(); ++i) {
    for (int j = 0; j < i + 1; ++j) result += "a";
    result += std::to_string(i);
  }
  for (int i = 0; i < world.rank() + 1; ++i) str += "a";
  str += std::to_string(world.rank());

  // gather strings
  for (int root = 0; root < world.size(); ++root) {
    auto str_gathered = mpi::gather(str, world, root);
    if (world.rank() == root)
      EXPECT_EQ(str_gathered, result);
    else
      EXPECT_TRUE(str_gathered.empty());
  }

  // allgather strings
  auto str_gathered = mpi::all_gather(str);
  EXPECT_EQ(str_gathered, result);

  // allgather empty strings
  auto empty_str = mpi::all_gather(std::string{});
  EXPECT_TRUE(empty_str.empty());
}

MPI_TEST_MAIN;
