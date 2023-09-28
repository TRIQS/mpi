// Copyright (c) 2023 Simons Foundation
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
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#include "mpi/mpi.hpp"
#include <gtest/gtest.h>
#include <numeric>

TEST(MPI_Window, SharedCommunicator) {
  mpi::communicator world;
  [[maybe_unused]] auto shm = world.split_shared();
}

TEST(MPI_Window, RingOneSidedGet) {
  mpi::communicator world;
  int const rank = world.rank();
  int const size = world.size();
  int const left = (rank-1+size) % size;

  int snd_buf, rcv_buf;
  mpi::window<int> win{world, &snd_buf, 1};
  snd_buf = rank;

  int sum = 0;
  for(int i = 0; i < size; ++i) {
    win.fence();
    win.get(&rcv_buf, 1, left);
    win.fence();
    snd_buf = rcv_buf;
    sum += rcv_buf;
  }

  EXPECT_EQ(sum, (size * (size - 1)) / 2);
}

TEST(MPI_Window, RingOneSidedPut) {
  mpi::communicator world;
  int const rank = world.rank();
  int const size = world.size();
  int const right = (rank+1) % size;

  int snd_buf, rcv_buf;
  mpi::window<int> win{world, &rcv_buf, 1};
  snd_buf = rank;

  int sum = 0;
  for(int i = 0; i < size; ++i) {
    win.fence();
    win.put(&snd_buf, 1, right);
    win.fence();
    snd_buf = rcv_buf;
    sum += rcv_buf;
  }

  EXPECT_EQ(sum, (size * (size - 1)) / 2);
}

TEST(MPI_Window, RingOneSidedAllowShared) {
  mpi::communicator world;
  auto shm = world.split_shared();
  int const rank_shm = shm.rank();
  int const size_shm = shm.size();
  int const right = (rank_shm+1) % size_shm;

  mpi::shared_window<int> win{shm, 1};
  int *rcv_buf_ptr = win.base(rank_shm);

  int snd_buf = rank_shm;
  int sum = 0;
  for(int i = 0; i < size_shm; ++i) {
      win.fence();
      win.put(&snd_buf, 1, right);
      win.fence();
      snd_buf = *rcv_buf_ptr;
      sum += *rcv_buf_ptr;
  }

  EXPECT_EQ(sum, (size_shm * (size_shm - 1)) / 2);
}

TEST(MPI_Window, SharedArray) {
  mpi::communicator world;
  auto shm = world.split_shared();
  int const rank_shm = shm.rank();

  constexpr int const size = 20;
  constexpr int const magic = 21;

  mpi::shared_window<int> win{shm, rank_shm == 0 ? size : 0};
  std::span array_view{win.base(0), static_cast<std::size_t>(win.size(0))};

  win.fence();
  if (rank_shm == 0) {
      for (auto &&el : array_view) {
          el = magic;
      }
  }
  win.fence();

  int sum = std::accumulate(array_view.begin(), array_view.end(), int{0});
  EXPECT_EQ(sum, size * magic);
}

MPI_TEST_MAIN;
