// Copyright (c) 2020 Simons Foundation
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
#include <itertools/itertools.hpp>
#include <itertools/omp_chunk.hpp>
#include <mpi/mpi.hpp>

#include <iostream>
#include <numeric>

using namespace itertools;

TEST(MPI, ChunkSingleIntegerRange) {
  // divide a range among the ranks, sum up the subranges and reduce the sum
  mpi::communicator world;
  for (auto n : range(10)) {
    auto sub_rg = mpi::chunk(range(n), world);
    auto sum    = std::accumulate(sub_rg.begin(), sub_rg.end(), 0l);
    sum         = mpi::all_reduce(sum, world);
    EXPECT_EQ(n * (n - 1) / 2, sum);
  }
}

TEST(MPI_Chunk, ChunkZippedIntegerRanges) {
  // zip two ranges, divide the zipped range among the ranks, sum up the subranges and reduce the sum
  mpi::communicator world;
  for (auto n : range(10)) {
    auto rg1 = range(0, n);
    auto rg2 = range(n, 2 * n);
    long sum = 0;
    for (auto [v1, v2] : mpi::chunk(zip(rg1, rg2), world)) { sum += v1 + v2; }
    sum = mpi::all_reduce(sum, world);
    EXPECT_EQ(n * (2 * n - 1), sum);
  }
}

TEST(MPI, OMPHybrid) {
  // first divide a range among MPI processes and then among OMP threads
  mpi::communicator world;
  int const n = 10;
  long sum    = 0;
#pragma omp parallel
  for (auto i : omp_chunk(mpi::chunk(range(n)))) {
#pragma omp critical
    {
      std::cout << "MPI rank: " << world.rank() << ", OMP thread: " << omp_get_thread_num() << ", Element: " << i << std::endl;
      sum += i;
    }
  }

  // reduce and check the sum, i.e. that every element of the range has been visited
  sum = mpi::all_reduce(sum, world);
  EXPECT_EQ(n * (n - 1) / 2, sum);
}

MPI_TEST_MAIN;
