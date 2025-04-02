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

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test reducing a range of objects.
template <typename T> void test_reduce_range(std::array<T, 5> const &values, std::array<T, 5> const &result, MPI_Op op = MPI_SUM) {
  mpi::communicator world;

  // reduce from different roots
  for (int root = 0; root < world.size(); ++root) {
    // reduce a span into an array
    auto arr = values;
    mpi::reduce_range(std::span{values.data() + 2, 3}, std::span{arr.begin(), 3}, world, root, false, op);
    if (world.rank() == root) {
      expect_range_eq(std::span{arr.data(), 3}, std::span{result.data() + 2, 3});
      expect_range_eq(std::span{arr.data() + 3, 2}, std::span{values.data() + 3, 2});
    } else {
      expect_range_eq(arr, values);
    }

    // reduce a list into a list
    std::list<T> list(values.begin(), values.end()), list_red(values.begin(), values.end());
    if (world.rank() == root) {
      mpi::reduce_range(list, list_red, world, root, false, op);
      expect_range_eq(list_red, result);
    } else {
      list_red.clear();
      mpi::reduce_range(list, list_red, world, root, false, op);
      EXPECT_TRUE(list_red.empty());
    }

    // reduce a view on a list in place
    list.assign(values.begin(), values.end());
    mpi::reduce_range(std::ranges::take_view(list, 2), std::ranges::take_view(list, 2), world, root, false, op);
    if (world.rank() == root) {
      expect_range_eq(std::ranges::take_view(list, 2), std::ranges::take_view(result, 2));
      expect_range_eq(std::ranges::drop_view(list, 2), std::ranges::drop_view(values, 2));
    } else {
      expect_range_eq(list, values);
    }

    // reduce a span in place
    arr = values;
    mpi::reduce_range(std::span{arr.data() + 2, 3}, std::span{arr.data() + 2, 3}, world, root, false, op);
    if (world.rank() == root) {
      expect_range_eq(std::span{arr.data() + 2, 3}, std::span{result.data() + 2, 3});
      expect_range_eq(std::span{arr.data(), 2}, std::span{values.data(), 2});
    } else {
      expect_range_eq(arr, values);
    }

    // reduce an array into a list
    if (world.rank() == root) {
      list = std::list<T>(5);
      mpi::reduce_range(values, list, world, root, false, op);
      expect_range_eq(list, result);
    } else {
      list.clear();
      mpi::reduce_range(values, list, world, root, false, op);
      EXPECT_TRUE(list.empty());
    }
  }

  // allreduce a list in place using reduce_range
  std::list<T> list(values.begin(), values.end());
  mpi::reduce_range(list, list, world, 0, true, op);
  expect_range_eq(list, result);

  // allreduce a span in place
  auto arr = values;
  mpi::reduce_range(std::span{arr.data() + 1, 3}, std::span{arr.data() + 1, 3}, world, 0, true, op);
  expect_range_eq(std::span{arr.data() + 1, 3}, std::span{result.data() + 1, 3});
  EXPECT_EQ(arr[0], values[0]);
  EXPECT_EQ(arr[4], values[4]);
}

TEST(MPI, ReduceIntegerRange) {
  mpi::communicator world;
  std::array<int, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = (i + 1) * (world.rank() + 1);
    result[i] = (i + 1) * world.size() * (world.size() + 1) / 2;
  }
  test_reduce_range(values, result);
}

TEST(MPI, ReduceComplexRange) {
  mpi::communicator world;
  double rank     = world.rank() + 1.0;
  double red_rank = world.size() * (world.size() + 1) * 0.5;
  std::array<std::complex<double>, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = std::complex<double>{rank * (i + 1), -rank * (i + 1)};
    result[i] = std::complex<double>{red_rank * (i + 1), -red_rank * (i + 1)};
  }
  test_reduce_range(values, result);
}

TEST(MPI, ReduceCustomMPITypeRange) {
  mpi::communicator world;
  long rank     = world.rank() + 1;
  long red_rank = world.size() * (world.size() + 1) / 2;
  std::array<mpi_t, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = mpi_t{rank * (i + 1)};
    result[i] = mpi_t{red_rank * (i + 1)};
  }
  if (world.size() > 1) { test_reduce_range(values, result, mpi::map_add<mpi_t>()); }
}

TEST(MPI, ReduceCustomNonMPITypeRange) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  std::array<non_mpi_t, 5> values{}, result{};
  for (int i = 0; i < 5; ++i) {
    values[i] = non_mpi_t{rank * (i + 1)};
    result[i] = non_mpi_t{red_rank * (i + 1)};
  }
  test_reduce_range(values, result);
}

MPI_TEST_MAIN;
