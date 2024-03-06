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

#include <mpi/mpi.hpp>
#include <mpi/vector.hpp>
#include <gtest/gtest.h>
#include <numeric>

// Test cases are adapted from slides and exercises of the HLRS course:
// Introduction to the Message Passing Interface (MPI)
// Authors: Joel Malard, Alan Simpson, (EPCC)
//          Rolf Rabenseifner, Traugott Streicher, Tobias Haas (HLRS)
// https://fs.hlrs.de/projects/par/par_prog_ws/pdf/mpi_3.1_rab.pdf
// https://fs.hlrs.de/projects/par/par_prog_ws/practical/MPI31single.tar.gz

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

TEST(MPI_Window, RingOneSidedAllocShared) {
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

TEST(MPI_Window, RingOneSidedStoreWinAllocSharedSignal) {
  mpi::communicator world;
  auto shm = world.split_shared();

  int const rank_shm = shm.rank();
  int const size_shm = shm.size();
  int const right = (rank_shm+1) % size_shm;
  int const left = (rank_shm-1+size_shm) % size_shm;

  mpi::shared_window<int> win{shm, 1};
  int *rcv_buf_ptr = win.base(rank_shm);
  win.lock();

  int sum = 0;
  int snd_buf = rank_shm;

  MPI_Request rq;
  MPI_Status status;
  int snd_dummy, rcv_dummy;

  for(int i = 0; i < size_shm; ++i) {
    // ... The local Win_syncs are needed to sync the processor and real memory.
    // ... The following pair of syncs is needed that the read-write-rule is fulfilled.
    win.sync();

    // ... tag=17: posting to left that rcv_buf is exposed to left, i.e.,
    //             the left process is now allowed to store data into the local rcv_buf
    MPI_Irecv(&rcv_dummy, 0, MPI_INT, right, 17, shm.get(), &rq);
    MPI_Send (&snd_dummy, 0, MPI_INT, left,  17, shm.get());
    MPI_Wait(&rq, &status);

    win.sync();

    // MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
    //   ... is substited by (with offset "right-my_rank" to store into right neigbor's rcv_buf):
    *(rcv_buf_ptr+(right-rank_shm)) = snd_buf;


    // ... The following pair of syncs is needed that the write-read-rule is fulfilled.
    win.sync();

    // ... The following communication synchronizes the processors in the way
    //     that the origin processor has finished the store
    //     before the target processor starts to load the data.
    // ... tag=18: posting to right that rcv_buf was stored from left
    MPI_Irecv(&rcv_dummy, 0, MPI_INT, left,  18, shm.get(), &rq);
    MPI_Send (&snd_dummy, 0, MPI_INT, right, 18, shm.get());
    MPI_Wait(&rq, &status);

    win.sync();

    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  EXPECT_EQ(sum, (size_shm * (size_shm - 1)) / 2);

  win.unlock();
}

TEST(MPI_Window, SharedArray) {
  mpi::communicator world;
  auto shm = world.split_shared();
  int const rank_shm = shm.rank();

  constexpr int const array_size = 20;
  constexpr int const magic = 21;

  mpi::shared_window<int> win{shm, rank_shm == 0 ? array_size : 0};
  std::span array_view{win.base(0), static_cast<std::size_t>(win.size(0))};

  win.fence();
  if (rank_shm == 0) {
      for (auto &&el : array_view) {
          el = magic;
      }
  }
  win.fence();

  int sum = std::accumulate(array_view.begin(), array_view.end(), int{0});
  EXPECT_EQ(sum, array_size * magic);
}

TEST(MPI_Window, DistributedSharedArray) {
  mpi::communicator world;
  auto shm = world.split_shared();

  // Number of total array elements (prime number to make it a bit more exciting)
  constexpr int const array_size_total = 197;

  // Create a communicator between rank0 of all shared memory islands ("head node")
  auto head = world.split(shm.rank() == 0 ? 0 : MPI_UNDEFINED);

  // Determine number of shared memory islands and broadcast to everyone
  int head_size = (world.rank() == 0 ? head.size(): -1);
  mpi::broadcast(head_size, world);

  // Determine rank in head node communicator and broadcast to all other ranks
  // on the same shared memory island
  int head_rank = (head.get() != MPI_COMM_NULL ? head.rank() : -1);
  mpi::broadcast(head_rank, shm);

  // Determine number of ranks on each shared memory island and broadcast to everyone
  std::vector<int> shm_sizes(head_size, 0);
  if (!head.is_null()) {
      shm_sizes.at(head_rank) = shm.size();
      shm_sizes = mpi::all_reduce(shm_sizes, head);
  }
  mpi::broadcast(shm_sizes, world);

  // Chunk the total array such that each rank has approximately the same number
  // of array elements
  std::vector<int> array_sizes(head_size, 0);
  for (auto &&[shm_size, array_size]: itertools::zip(shm_sizes, array_sizes)) {
      array_size = array_size_total / world.size() * shm_size;
  }
  // Distribute the remainder evenly over the islands to reduce load imbalance
  for (auto i: itertools::range(array_size_total % world.size())) {
      array_sizes.at(i % array_sizes.size()) += 1;
  }

  EXPECT_EQ(array_size_total, std::accumulate(array_sizes.begin(), array_sizes.end(), int{0}));

  // Determine the global index offset on the current shared memory island
  auto begin = array_sizes.begin();
  std::advance(begin, head_rank);
  std::ptrdiff_t offset = std::accumulate(array_sizes.begin(), begin, std::ptrdiff_t{0});

  // Allocate memory
  mpi::shared_window<int> win{shm, shm.rank() == 0 ? array_sizes.at(head_rank) : 0};
  std::span array_view{win.base(0), static_cast<std::size_t>(win.size(0))};

  // Fill array with global index (= local index + global offset)
  // We do this in parallel on each shared memory island by chunking the total range
  win.fence();
  auto slice = itertools::chunk_range(0, array_view.size(), shm.size(), shm.rank());
  for (auto i = slice.first; i < slice.second; ++i) {
      array_view[i] = i + offset;
  }
  win.fence();

  // Calculate partial sum on head node of each shared memory island and
  // all_reduce the partial sums into a total sum over the head node
  // communicator and broadcast result to everyone
  std::vector<int> partial_sum(head_size, 0);
  int sum = 0;
  if (!head.is_null()) {
      partial_sum[head_rank] = std::accumulate(array_view.begin(), array_view.end(), int{0});
      partial_sum = mpi::all_reduce(partial_sum, head);
      sum = std::accumulate(partial_sum.begin(), partial_sum.end(), int{0});
  }
  mpi::broadcast(sum, world);

  // Total sum is just sum of numbers in interval [0, array_size_total)
  EXPECT_EQ(sum, (array_size_total * (array_size_total - 1)) / 2);
}

MPI_TEST_MAIN;
