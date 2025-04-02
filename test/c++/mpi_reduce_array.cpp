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

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test reducing arrays.
template <typename T> void test_reduce_array(std::array<T, 5> const &values, std::array<T, 5> const &result, MPI_Op op = MPI_SUM) {
  mpi::communicator world;

  // reduce from different roots
  for (int root = 0; root < world.size(); ++root) {
    // reduce an array into a new array
    auto arr = mpi::reduce(values, world, root, false, op);
    if (world.rank() == root) expect_range_eq(arr, result);

    // reduce an empty array
    std::array<T, 0> empty_arr{};
    auto empty_red = mpi::reduce(empty_arr, world, root, false, op);
    static_assert(empty_red.size() == 0);

    // reduce an array in place
    arr = values;
    mpi::reduce_in_place(arr, world, root, false, op);
    if (world.rank() == root)
      expect_range_eq(arr, result);
    else
      expect_range_eq(arr, values);

    // reduce an array into an existing array
    arr = {};
    mpi::reduce_into(values, arr, world, root, false, op);
    if (world.rank() == root) expect_range_eq(arr, result);

    // reduce an empty array into an existing array
    mpi::reduce_into(empty_arr, empty_arr, world, root, false, op);
  }

  // allreduce an array into new array
  auto arr = mpi::all_reduce(values, world, op);
  expect_range_eq(arr, result);

  // allreduce an array in place
  arr = values;
  mpi::all_reduce_in_place(arr, world, op);
  expect_range_eq(arr, result);

  // allreduce an array in place using all_reduce_into
  arr = values;
  mpi::all_reduce_into(arr, arr, world, op);
  expect_range_eq(arr, result);
}

TEST(MPI, ReduceIntegerArray) {
  mpi::communicator world;
  std::array<int, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = (i + 1) * (world.rank() + 1);
    result[i] = (i + 1) * world.size() * (world.size() + 1) / 2;
  }
  test_reduce_array(values, result);
}

TEST(MPI, ReduceComplexArray) {
  mpi::communicator world;
  double rank     = world.rank() + 1.0;
  double red_rank = world.size() * (world.size() + 1) * 0.5;
  std::array<std::complex<double>, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = std::complex<double>{rank * (i + 1), -rank * (i + 1)};
    result[i] = std::complex<double>{red_rank * (i + 1), -red_rank * (i + 1)};
  }
  test_reduce_array(values, result);
}

TEST(MPI, ReduceCustomMPITypeArray) {
  mpi::communicator world;
  long rank     = world.rank() + 1;
  long red_rank = world.size() * (world.size() + 1) / 2;
  std::array<mpi_t, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = mpi_t{rank * (i + 1)};
    result[i] = mpi_t{red_rank * (i + 1)};
  }
  if (world.size() > 1) { test_reduce_array(values, result, mpi::map_add<mpi_t>()); }
}

TEST(MPI, ReduceCustomNonMPITypeArray) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  std::array<non_mpi_t, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = non_mpi_t{rank * (i + 1)};
    result[i] = non_mpi_t{red_rank * (i + 1)};
  }
  test_reduce_array(values, result);
}

MPI_TEST_MAIN;
