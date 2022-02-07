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

#include <mpi/mpi.hpp>

#include <gtest/gtest.h>
#include <numeric>

using namespace itertools;

TEST(MpiChunk, Simple) {

  mpi::communicator comm{};

  for (int N : range(10)) {
    auto chunk_V = mpi::chunk(range(N), comm);
    int sum      = std::accumulate(chunk_V.begin(), chunk_V.end(), 0);
    sum          = mpi::all_reduce(sum, comm);
    EXPECT_EQ(N * (N - 1) / 2, sum);
  }
}

TEST(MpiChunk, Multi) {

  mpi::communicator comm{};

  for (int N : range(10)) {
    auto V1 = range(0, N);
    auto V2 = range(N, 2 * N);
    int sum = 0;
    for (auto [v1, v2] : mpi::chunk(zip(V1, V2), comm)) { sum += v1 + v2; }
    sum = mpi::all_reduce(sum, comm);
    EXPECT_EQ(N * (2 * N - 1), sum);
  }
}

#include <itertools/omp_chunk.hpp>

TEST(MpiChunk, OMPHybrid) {

  int const N = 10;

  mpi::communicator comm{};
  int sum = 0;

#pragma omp parallel
  for (auto i : omp_chunk(mpi::chunk(range(N)))) {
#pragma omp critical
    {
      std::cout << "mpi_rank " << comm.rank() << "  omp_thread " << omp_get_thread_num() << "  i " << i << std::endl;
      sum += i;
    }
  }

  sum = mpi::all_reduce(sum, comm);
  EXPECT_EQ(N * (N - 1) / 2, sum);
}

MPI_TEST_MAIN;
