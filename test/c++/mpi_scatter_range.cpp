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
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

// Check if two ranges are equal.
void expect_range_eq(auto &&rg1, auto &&rg2) {
  EXPECT_EQ(std::ranges::size(rg1), std::ranges::size(rg2));
  auto it2 = std::ranges::begin(rg2);
  for (auto &&a : rg1) { EXPECT_EQ(a, *it2++); }
}

// Test scattering a vector.
template <typename T> void test_scatter_range(std::vector<T> const &values, long chunk_size) {
  mpi::communicator world;
  const int rank = world.rank();
  auto sizes     = std::vector<int>(world.size());
  for (int i = 0; i < world.size(); ++i) sizes[i] = static_cast<int>(mpi::chunk_length(values.size(), world.size(), i, chunk_size));
  auto acc_sizes = std::vector<int>(world.size() + 1, 0);
  std::partial_sum(sizes.begin(), sizes.end(), std::next(acc_sizes.begin()));
  EXPECT_EQ(acc_sizes.back(), values.size());

  // scatter from different roots
  for (int root = 0; root < world.size(); ++root) {
    // scatter a vector into a span
    auto vec = std::vector<T>(sizes[rank], T{0});
    mpi::scatter_range(values, std::span(vec.begin(), sizes[rank]), values.size(), world, root, chunk_size);
    expect_range_eq(vec, std::span(values.begin() + acc_sizes[rank], sizes[rank]));

    // scatter with chunk size = number of elements to be scattered
    vec = std::vector<T>((rank == 0 ? values.size() : 0), T{0});
    mpi::scatter_range(values, vec, values.size(), world, root, values.size());
    if (world.rank() == 0)
      expect_range_eq(vec, values);
    else
      EXPECT_TRUE(vec.empty());
  }
}

TEST(MPI, ScatterIntegerRange) {
  mpi::communicator world;
  const long min_nchunks = 3;
  const long chunk_size  = 4;
  for (int i = 0; i < world.size(); ++i) {
    // chunk size = 1
    std::vector<int> values(min_nchunks * world.size() + i);
    std::iota(values.begin(), values.end(), 0);
    test_scatter_range(values, 1);

    // chunk size = 4
    values.resize((min_nchunks * world.size() + i) * chunk_size);
    std::iota(values.begin(), values.end(), 0);
    test_scatter_range(values, chunk_size);
  }
}

TEST(MPI, ScatterComplexRange) {
  mpi::communicator world;
  const long min_nchunks = 3;
  const long chunk_size  = 4;
  for (int i = 0; i < world.size(); ++i) {
    // chunk size = 1
    std::vector<std::complex<double>> values(min_nchunks * world.size() + i);
    for (int j = 0; j < values.size(); ++j) values[j] = std::complex<double>(j, -j);
    test_scatter_range(values, 1);

    // chunk size = 4
    values.resize((min_nchunks * world.size() + i) * chunk_size);
    for (int j = 0; j < values.size(); ++j) values[j] = std::complex<double>(j, -j);
    test_scatter_range(values, chunk_size);
  }
}

TEST(MPI, ScatterCustomMPITypeRange) {
  mpi::communicator world;
  const long min_nchunks = 3;
  const long chunk_size  = 4;
  for (int i = 0; i < world.size(); ++i) {
    // chunk size = 1
    std::vector<mpi_t> values(min_nchunks * world.size() + i);
    for (int j = 0; j < values.size(); ++j) values[j].a = j;
    test_scatter_range(values, 1);

    // chunk size = 4
    values.resize((min_nchunks * world.size() + i) * chunk_size);
    for (int j = 0; j < values.size(); ++j) values[j].a = j;
    test_scatter_range(values, chunk_size);
  }
}

MPI_TEST_MAIN;
