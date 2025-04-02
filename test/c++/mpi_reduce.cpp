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
#include <utility>

// Test reducing a single value/object.
template <typename T> void test_reduce(T value, T result, T def_value, MPI_Op op = MPI_SUM) {
  mpi::communicator world;

  // reduce from different roots
  for (int root = 0; root < world.size(); ++root) {
    // reduce an object into new object
    auto red_value = mpi::reduce(value, world, root, false, op);
    if (world.rank() == root) { EXPECT_EQ(red_value, result); }

    // reduce an object in place
    red_value = value;
    mpi::reduce_in_place(red_value, world, root, false, op);
    if (world.rank() == root)
      EXPECT_EQ(red_value, result);
    else
      EXPECT_EQ(red_value, value);

    // reduce an object into an existing object
    red_value = def_value;
    mpi::reduce_into(value, red_value, world, root, false, op);
    if (world.rank() == root)
      EXPECT_EQ(red_value, result);
    else
      EXPECT_EQ(red_value, def_value);
  }

  // allreduce an object into a new object
  auto red_value = mpi::all_reduce(value, world, op);
  EXPECT_EQ(red_value, result);

  // allreduce an object in place
  red_value = value;
  mpi::all_reduce_in_place(red_value, world, op);
  EXPECT_EQ(red_value, result);

  // allreduce an object using all_reduce_into
  red_value = value;
  mpi::all_reduce_into(value, red_value, world, op);
  EXPECT_EQ(red_value, result);

  // allreduce an object in place using all_reduce_into
  red_value = value;
  mpi::all_reduce_into(red_value, red_value, world, op);
  EXPECT_EQ(red_value, result);
}

TEST(MPI, ReduceInteger) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  test_reduce(rank, red_rank, 0);
}

TEST(MPI, ReduceComplex) {
  mpi::communicator world;
  double rank     = world.rank() + 1.0;
  double red_rank = world.size() * (world.size() + 1) * 0.5;
  test_reduce(std::complex<double>{rank, -rank}, std::complex<double>{red_rank, -red_rank}, std::complex<double>{0, 0});
}

TEST(MPI, ReduceCustomMPIType) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  if (world.size() > 1) test_reduce(mpi_t{rank}, mpi_t{red_rank}, mpi_t{0}, mpi::map_add<mpi_t>());
}

TEST(MPI, ReduceCustomNonMPIType) {
  mpi::communicator world;
  int rank     = world.rank() + 1;
  int red_rank = world.size() * (world.size() + 1) / 2;
  test_reduce(non_mpi_t{rank}, non_mpi_t{red_rank}, non_mpi_t{0});
}

// Test reducing a pair.
TEST(MPI, ReducePair) {
  mpi::communicator world;

  // allreduce a pair of integers
  auto p1 = mpi::all_reduce(std::pair{world.rank(), -world.rank()}, world, MPI_MAX);
  EXPECT_EQ(p1.first, world.size() - 1);
  EXPECT_EQ(p1.second, 0);

  // reduce a pair of non_mpi_t
  auto p2 = mpi::reduce(std::pair{non_mpi_t{1}, non_mpi_t{world.rank() + 1}}, world, world.size() - 1);
  if (world.rank() == world.size() - 1) {
    EXPECT_EQ(p2.first, non_mpi_t(world.size()));
    EXPECT_EQ(p2.second, non_mpi_t(world.size() * (world.size() + 1) / 2));
  } else {
    EXPECT_EQ(p2.first, non_mpi_t());
    EXPECT_EQ(p2.second, non_mpi_t());
  }
}

MPI_TEST_MAIN;
