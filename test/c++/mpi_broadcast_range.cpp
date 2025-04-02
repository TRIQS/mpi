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
#include <list>
#include <ranges>
#include <span>
#include <utility>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test broadcasting a range of objects.
template <typename T> void test_broadcast_range(std::array<T, 5> root_values) {
  mpi::communicator world;
  std::array<T, 5> def_arr{};
  def_arr.fill(root_values[0]);

  // broadcast a contiguous range from different roots
  auto arr = root_values;
  for (int root = 0; root < world.size(); ++root) {
    if (world.rank() == root) {
      arr = root_values;
      mpi::broadcast_range(std::span{arr.begin() + 2, 3}, world, root);
      expect_range_eq(arr, root_values);
    } else {
      arr = def_arr;
      mpi::broadcast_range(std::span{arr.begin(), 3}, world, root);
      expect_range_eq(std::span{arr.begin(), 3}, std::span{root_values.begin() + 2, 3});
      expect_range_eq(std::span{arr.begin() + 3, 2}, std::span{def_arr.begin() + 3, 2});
    }
  }

  // broadcast a view on a non-contiguous list
  std::list<T> list(def_arr.begin(), def_arr.end());
  if (world.rank() == 0) list.assign(root_values.begin(), root_values.end());
  mpi::broadcast_range(std::ranges::drop_view(list, 2), world);
  if (world.rank() == 0) {
    expect_range_eq(list, root_values);
  } else {
    expect_range_eq(std::ranges::drop_view(list, 2), std::ranges::drop_view(root_values, 2));
    expect_range_eq(std::ranges::take_view(list, 2), std::ranges::take_view(def_arr, 2));
  }
}

TEST(MPI, BroadcastIntegerRange) { test_broadcast_range(std::array<int, 5>{1, 2, 3, 4, 5}); }

TEST(MPI, BroadcastComplexRange) {
  using namespace std::complex_literals;
  test_broadcast_range(std::array<std::complex<double>, 5>{1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i, 5.0 - 5.0i});
}

TEST(MPI, BroadcastCustomMPITypeRange) { test_broadcast_range(std::array<mpi_t, 5>{mpi_t{1}, mpi_t{2}, mpi_t{3}, mpi_t{4}, mpi_t{5}}); }

TEST(MPI, BroadcastCustomNonMPITypeRange) {
  test_broadcast_range(std::array<non_mpi_t, 5>{non_mpi_t{1}, non_mpi_t{2}, non_mpi_t{3}, non_mpi_t{4}, non_mpi_t{5}});
}

TEST(MPI, BroadcastStringRange) { test_broadcast_range(std::array<std::string, 5>{"Hello", "World", "MPI", "Broadcast", "Array"}); }

TEST(MPI, BroadcastPairRange) {
  test_broadcast_range(std::array<std::pair<int, std::string>, 5>{{{1, "Hello"}, {2, "World"}, {3, "MPI"}, {4, "Broadcast"}, {5, "Array"}}});
}

MPI_TEST_MAIN;
