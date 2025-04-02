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

// Test reducing a vector.
template <typename T> void test_reduce_vector(std::vector<T> const &values, std::vector<T> const &result, MPI_Op op = MPI_SUM) {
  mpi::communicator world;

  // reduce from different roots
  for (int root = 0; root < world.size(); ++root) {
    // reduce a vector into a new vector
    auto vec = mpi::reduce(values, world, root, false, op);
    if (world.rank() == root) expect_range_eq(vec, result);

    // reduce an empty vector
    auto empty_vec = mpi::reduce(std::vector<T>{}, world, root, false, op);
    EXPECT_EQ(empty_vec.size(), 0);

    // reduce a vector in place
    vec = values;
    mpi::reduce_in_place(vec, world, root, false, op);
    if (world.rank() == root)
      expect_range_eq(vec, result);
    else
      expect_range_eq(vec, values);

    // reduce an empty vector in place
    mpi::reduce_in_place(empty_vec, world, root, false, op);
    EXPECT_EQ(empty_vec.size(), 0);

    // reduce a vector into an existing empty vector
    vec.clear();
    mpi::reduce_into(values, vec, world, root, false, op);
    if (world.rank() == root)
      expect_range_eq(vec, result);
    else
      EXPECT_TRUE(vec.empty());

    // reduce an empty vector into an existing vector
    vec = values;
    mpi::reduce_into(empty_vec, vec, world, root, false, op);
    if (world.rank() == root)
      EXPECT_EQ(vec.size(), 0);
    else
      expect_range_eq(vec, values);
  }

  // allreduce a vector into a new vector
  auto vec = mpi::all_reduce(values, world, op);
  expect_range_eq(vec, result);

  // allreduce a vector in place
  vec = values;
  mpi::all_reduce_in_place(vec, world, op);
  expect_range_eq(vec, result);

  // allreduce a vector in place using all_reduce_into
  vec = values;
  mpi::all_reduce_into(vec, vec, world, op);
  expect_range_eq(vec, result);
}

TEST(MPI, ReduceIntegerVector) {
  mpi::communicator world;
  std::vector<int> values(5), result(5);
  for (int i = 0; i < 5; ++i) {
    values[i] = (i + 1) * (world.rank() + 1);
    result[i] = (i + 1) * world.size() * (world.size() + 1) / 2;
  }
  test_reduce_vector(values, result);
}

TEST(MPI, ReduceComplexVector) {
  mpi::communicator world;
  double rank     = world.rank() + 1.0;
  double red_rank = world.size() * (world.size() + 1) * 0.5;
  std::vector<std::complex<double>> values(5), result(5);
  for (int i = 0; i < 5; ++i) {
    values[i] = std::complex<double>{rank * (i + 1), -rank * (i + 1)};
    result[i] = std::complex<double>{red_rank * (i + 1), -red_rank * (i + 1)};
  }
  test_reduce_vector(values, result);
}

TEST(MPI, ReduceCustomMPITypeVector) {
  mpi::communicator world;
  long rank     = world.rank() + 1;
  long red_rank = world.size() * (world.size() + 1) / 2;
  std::vector<mpi_t> values(5), result(5);
  for (int i = 0; i < 5; ++i) {
    values[i] = mpi_t{rank * (i + 1)};
    result[i] = mpi_t{red_rank * (i + 1)};
  }
  if (world.size() > 1) { test_reduce_vector(values, result, mpi::map_add<mpi_t>()); }
}

TEST(MPI, ReduceCustomNonMPITypeVector) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  std::vector<non_mpi_t> values(5), result(5);
  for (int i = 0; i < 5; ++i) {
    values[i] = non_mpi_t{rank * (i + 1)};
    result[i] = non_mpi_t{red_rank * (i + 1)};
  }
  test_reduce_vector(values, result);
}

MPI_TEST_MAIN;
