// Copyright (c) 2024 Simons Foundation
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
// Authors: Nils Wentzell

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <complex>
#include <optional>

// Test broadcasting an optional with a value.
template <typename T> void test_broadcast_optional(T root_value) {
  mpi::communicator world;
  for (int root = 0; root < world.size(); ++root) {
    std::optional<T> bcast_value{};
    if (world.rank() == root) bcast_value = root_value;
    mpi::broadcast(bcast_value, world, root);
    EXPECT_EQ(bcast_value, root_value);
  }
}

TEST(MPI, BroadcastOptionalInt) { test_broadcast_optional(42); }

TEST(MPI, BroadcastOptionalComplex) { test_broadcast_optional(std::complex<double>{1.0, 2.0}); }

TEST(MPI, BroadcastEmptyOptionalInt) {
  mpi::communicator world;
  for (int root = 0; root < world.size(); ++root) {
    std::optional<int> bcast_value{};
    if (world.rank() != root) bcast_value = {}; // non-root has value, root is empty
    mpi::broadcast(bcast_value, world, root);
    EXPECT_FALSE(bcast_value.has_value()); // after broadcast, all should be empty like root
  }
}

// Test reducing an optional with a value.
template <typename T> void test_reduce_optional(T value, T result) {
  mpi::communicator world;

  // reduce from different roots
  for (int root = 0; root < world.size(); ++root) {
    // reduce into new object
    auto red_value = mpi::reduce(std::optional{value}, world, root);
    if (world.rank() == root) { EXPECT_EQ(red_value, result); }

    // reduce into existing object
    std::optional<T> red_out{};
    mpi::reduce_into(std::optional{value}, red_out, world, root);
    if (world.rank() == root) { EXPECT_EQ(red_out, result); }
  }

  // allreduce into new object
  auto red_value = mpi::all_reduce(std::optional{value}, world);
  EXPECT_EQ(red_value, result);

  // allreduce into existing object
  std::optional<T> red_out{};
  mpi::all_reduce_into(std::optional{value}, red_out, world);
  EXPECT_EQ(red_out, result);
}

TEST(MPI, ReduceOptionalInt) {
  mpi::communicator world;
  int result = world.size() * (world.size() - 1) / 2;
  test_reduce_optional(world.rank(), result);
}

TEST(MPI, ReduceOptionalComplex) {
  mpi::communicator world;
  double rank   = world.rank();
  double result = world.size() * (world.size() - 1) * 0.5;
  test_reduce_optional(std::complex<double>{rank, -rank}, std::complex<double>{result, -result});
}

TEST(MPI, ReduceEmptyOptionalInt) {
  mpi::communicator world;

  for (int root = 0; root < world.size(); ++root) {
    auto red_value = mpi::reduce(std::optional<int>{}, world, root);
    EXPECT_FALSE(red_value.has_value());

    std::optional<int> red_out{};
    mpi::reduce_into(std::optional<int>{}, red_out, world, root);
    EXPECT_FALSE(red_out.has_value());
  }

  auto red_value = mpi::all_reduce(std::optional<int>{}, world);
  EXPECT_FALSE(red_value.has_value());
}

MPI_TEST_MAIN;
