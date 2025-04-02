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
#include <string>
#include <utility>

// Test broadcasting a single value/object.
template <typename T> void test_broadcast(T root_value) {
  mpi::communicator world;
  for (int root = 0; root < world.size(); ++root) {
    T bcast_value{};
    if (world.rank() == root) bcast_value = root_value;
    mpi::broadcast(bcast_value, world, root);
    EXPECT_EQ(bcast_value, root_value);
  }
}

TEST(MPI, BroadcastInteger) { test_broadcast(42); }

TEST(MPI, BroadcastComplex) { test_broadcast(std::complex<double>{1.0, 2.0}); }

TEST(MPI, BroadcastCustomMPIType) { test_broadcast(mpi_t{42}); }

TEST(MPI, BroadcastCustomNonMPIType) { test_broadcast(non_mpi_t{42}); }

TEST(MPI, BroadcastString) { test_broadcast(std::string{"Hello World"}); }

TEST(MPI, BroadcastPairOfStringAndComplex) { test_broadcast(std::make_pair(std::string{"Hello"}, std::complex<double>{1.0, 2.0})); }

TEST(MPI, BroadcastPairOfCustomMPITypeAndCustomNonMPIType) { test_broadcast(std::make_pair(mpi_t{42}, non_mpi_t{-5})); }

MPI_TEST_MAIN;
