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
template <typename T> void test_scatter_vector(std::vector<T> const &values) {
  mpi::communicator world;
  auto recvcounts = std::vector<int>(world.size());
  for (int i = 0; i < world.size(); ++i) recvcounts[i] = static_cast<int>(mpi::chunk_length(values.size(), world.size(), i));
  auto displs = std::vector<int>(world.size() + 1, 0);
  std::partial_sum(recvcounts.begin(), recvcounts.end(), std::next(displs.begin()));
  auto const recvcount = recvcounts[world.rank()];
  auto const displ     = displs[world.rank()];

  // scatter from different roots
  for (int root = 0; root < world.size(); ++root) {
    // scatter a vector into a new vector
    auto vec = mpi::scatter(world.rank() == root ? values : std::vector<T>{}, world, root);
    expect_range_eq(vec, std::span(values.begin() + displ, recvcount));

    // scatter a vector into an existing vector
    vec.clear();
    mpi::scatter_into(values, vec, world, root);
    expect_range_eq(vec, std::span(values.begin() + displ, recvcount));
  }

  // scatter an empty vector
  auto vec = mpi::scatter(std::vector<T>{}, world);
  EXPECT_TRUE(vec.empty());
}

TEST(MPI, ScatterIntegerVector) {
  mpi::communicator world;
  for (int total_size = 3 * world.size(); total_size < 4 * world.size(); ++total_size) {
    std::vector<int> values(total_size);
    std::iota(values.begin(), values.end(), 0);
    test_scatter_vector(values);
  }
}

TEST(MPI, ScatterComplexVector) {
  mpi::communicator world;
  for (int total_size = 3 * world.size(); total_size < 4 * world.size(); ++total_size) {
    std::vector<std::complex<double>> values(total_size);
    for (int i = 0; i < total_size; ++i) values[i] = std::complex<double>(i, -i);
    test_scatter_vector(values);
  }
}

TEST(MPI, ScatterCustomMPITypeVector) {
  mpi::communicator world;
  for (int total_size = 3 * world.size(); total_size < 4 * world.size(); ++total_size) {
    std::vector<mpi_t> values(total_size);
    for (int i = 0; i < total_size; ++i) values[i].a = i;
    test_scatter_vector(values);
  }
}

MPI_TEST_MAIN;
