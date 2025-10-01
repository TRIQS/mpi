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

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <array>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

// Test cases are adapted from slides and exercises of the HLRS course:
// Introduction to the Message Passing Interface (MPI)
// Authors: Joel Malard, Alan Simpson, (EPCC)
//          Rolf Rabenseifner, Traugott Streicher, Tobias Haas (HLRS)
// https://fs.hlrs.de/projects/par/par_prog_ws/pdf/mpi_3.1_rab.pdf
// https://fs.hlrs.de/projects/par/par_prog_ws/practical/MPI31single.tar.gz

TEST(MPI, WindowCommunicatorMember) {
  mpi::communicator world;

  int data = world.rank();

  mpi::window<int> win(world, &data, 1);

  auto win_comm = win.get_communicator();

  EXPECT_EQ(win_comm.rank(), world.rank());
  EXPECT_EQ(win_comm.size(), world.size());
}

TEST(MPI, WindowSharedCommunicatorMember) {
  auto shm = mpi::communicator{}.split_shared();

  mpi::shared_window<int> win{shm, 1};

  auto sh_win_comm = win.get_communicator();

  EXPECT_EQ(sh_win_comm.rank(), shm.rank());
  EXPECT_EQ(sh_win_comm.size(), shm.size());
}

TEST(MPI, WindowGetAttrBase) {
  mpi::communicator world;

  int buffer = world.rank();
  mpi::window<int> win{world, &buffer, 1};

  void *base_ptr = win.base();
  EXPECT_NE(base_ptr, nullptr);
  EXPECT_EQ(base_ptr, &buffer);
}

TEST(MPI, WindowAllocate) {
  mpi::communicator world;
  int rank = world.rank();

  mpi::window<int> win{world, 1};
  *(win.base()) = rank;

  win.fence();
  int rcv{};
  win.get(&rcv, 1, rank);
  win.fence();

  EXPECT_EQ(rcv, rank);
}

TEST(MPI, WindowPassiveTargetCommunication) {
  mpi::communicator world;
  if (world.size() < 2) { GTEST_SKIP() << "Test requires at least 2 processes\n"; }
  int rank = world.rank();

  auto win_comm = world.split(rank == 0 || rank == 1 ? 0 : MPI_UNDEFINED);

  if (rank == 0 || rank == 1) {
    mpi::window<int> win{win_comm, 1};
    *(win.base()) = -1;

    win.fence();
    if (rank == 0) {
      int val = 42;
      win.put(&val, 1, 1);
    }
    win.fence();

    if (rank == 1) { EXPECT_EQ(*(win.base()), 42); }
  }
}

TEST(MPI, WindowActiveTargetCommunication) {
  mpi::communicator world;
  if (world.size() < 2) {
    // Target rank cannot be equal to origin rank (deadlocks), so we need at
    // least two ranks for this test case.
    GTEST_SKIP();
  }
  int rank = world.rank();

  mpi::window<int> win{world, 1};
  *(win.base()) = -1;

  int origin_rank = 0;
  int target_rank = 1;

  // Only the origin and target ranks will participate in the communication.
  mpi::group world_group(world);
  auto origin_group = world_group.include({origin_rank});
  auto target_group = world_group.include({target_rank});

  if (rank == target_rank) {
    win.post(origin_group);
    win.wait(); // blocks until origin_rank calls complete()
    EXPECT_EQ(*(win.base()), 42);
  }

  if (rank == origin_rank) {
    win.start(target_group); // blocks until target_rank calls post()
    auto origin_arr  = std::array<int, 1>{42};
    int origin_count = 1;
    win.put(origin_arr.data(), origin_count, target_rank);
    win.complete();
  }
}

TEST(MPI, WindowGetAttrSize) {
  mpi::communicator world;
  int buffer{};
  mpi::window<int> win{world, &buffer, 1};

  MPI_Aint size = win.size();
  EXPECT_EQ(size, 1);
}

TEST(MPI, WindowMoveConstructor) {
  mpi::communicator world;
  int i = 1;
  mpi::window<int> win1{world, &i, 1};

  mpi::window<int> win2 = std::move(win1);

  EXPECT_EQ(win2.base(), &i);
  EXPECT_EQ(win1.base(), nullptr);
}

TEST(MPI, WindowNullptrSizeZero) {
  mpi::communicator world;
  mpi::window<int> win{world, nullptr, 0};

  EXPECT_EQ(win.base(), nullptr);
  EXPECT_EQ(win.size(), 0);
}

TEST(MPI, WindowOneSidedGet) {
  mpi::communicator world;
  int const rank = world.rank();

  int snd_buf{}, rcv_buf = -1;
  mpi::window<int> win{world, &snd_buf, 1};
  snd_buf = rank;

  win.fence();
  win.get(&rcv_buf, 1, rank);
  win.fence();

  EXPECT_EQ(rcv_buf, rank);
}

TEST(MPI, WindowOneSidedPut) {
  mpi::communicator world;
  int const rank = world.rank();

  int snd_buf{}, rcv_buf = -1;
  mpi::window<int> win{world, &rcv_buf, 1};
  snd_buf = rank;

  win.fence();
  win.put(&snd_buf, 1, rank);
  win.fence();

  EXPECT_EQ(rcv_buf, rank);
}

TEST(MPI, WindowRingOneSidedGet) {
  mpi::communicator world;
  int const rank = world.rank();
  int const size = world.size();
  int const left = (rank - 1 + size) % size;

  int snd_buf{}, rcv_buf{};
  mpi::window<int> win{world, &snd_buf, 1};
  snd_buf = rank;

  int sum = 0;
  for (int i = 0; i < size; ++i) {
    win.fence();
    win.get(&rcv_buf, 1, left);
    win.fence();
    snd_buf = rcv_buf;
    sum += rcv_buf;
  }

  EXPECT_EQ(sum, (size * (size - 1)) / 2);
}

TEST(MPI, WindowRingOneSidedPut) {
  mpi::communicator world;
  int const rank  = world.rank();
  int const size  = world.size();
  int const right = (rank + 1) % size;

  int snd_buf{}, rcv_buf{};
  mpi::window<int> win{world, &rcv_buf, 1};
  snd_buf = rank;

  int sum = 0;
  for (int i = 0; i < size; ++i) {
    win.fence();
    win.put(&snd_buf, 1, right);
    win.fence();
    snd_buf = rcv_buf;
    sum += rcv_buf;
  }

  EXPECT_EQ(sum, (size * (size - 1)) / 2);
}

TEST(MPI, WindowRingOneSidedAllocShared) {
  mpi::communicator world;
  auto shm           = world.split_shared();
  int const rank_shm = shm.rank();
  int const size_shm = shm.size();
  int const right    = (rank_shm + 1) % size_shm;

  mpi::shared_window<int> win{shm, 1};
  int *rcv_buf_ptr = win.base(rank_shm);

  int snd_buf = rank_shm;
  int sum     = 0;
  for (int i = 0; i < size_shm; ++i) {
    win.fence();
    win.put(&snd_buf, 1, right);
    win.fence();
    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  EXPECT_EQ(sum, (size_shm * (size_shm - 1)) / 2);
}

TEST(MPI, WindowRingOneSidedStoreWinAllocSharedSignal) {
  if (not mpi::has_env) {
    // Test doesn't make sense without MPI
    GTEST_SKIP();
  }
  mpi::communicator world;
  auto shm = world.split_shared();

  int const rank_shm = shm.rank();
  int const size_shm = shm.size();
  int const right    = (rank_shm + 1) % size_shm;
  int const left     = (rank_shm - 1 + size_shm) % size_shm;

  mpi::shared_window<int> win{shm, 1};
  int *rcv_buf_ptr = win.base(rank_shm);
  win.lock();

  int sum     = 0;
  int snd_buf = rank_shm;

  MPI_Request rq{};
  MPI_Status status;
  int snd_dummy{}, rcv_dummy{};

  for (int i = 0; i < size_shm; ++i) {
    // ... The local Win_syncs are needed to sync the processor and real memory.
    // ... The following pair of syncs is needed that the read-write-rule is fulfilled.
    win.sync();

    // ... tag=17: posting to left that rcv_buf is exposed to left, i.e.,
    //             the left process is now allowed to store data into the local rcv_buf
    MPI_Irecv(&rcv_dummy, 0, MPI_INT, right, 17, shm.get(), &rq);
    MPI_Send(&snd_dummy, 0, MPI_INT, left, 17, shm.get());
    MPI_Wait(&rq, &status);

    win.sync();

    // MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
    //   ... is substited by (with offset "right-my_rank" to store into right neigbor's rcv_buf):
    *(rcv_buf_ptr + (right - rank_shm)) = snd_buf;

    // ... The following pair of syncs is needed that the write-read-rule is fulfilled.
    win.sync();

    // ... The following communication synchronizes the processors in the way
    //     that the origin processor has finished the store
    //     before the target processor starts to load the data.
    // ... tag=18: posting to right that rcv_buf was stored from left
    MPI_Irecv(&rcv_dummy, 0, MPI_INT, left, 18, shm.get(), &rq);
    MPI_Send(&snd_dummy, 0, MPI_INT, right, 18, shm.get());
    MPI_Wait(&rq, &status);

    win.sync();

    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  EXPECT_EQ(sum, (size_shm * (size_shm - 1)) / 2);

  win.unlock();
}

TEST(MPI, WindowSharedArray) {
  mpi::communicator world;
  auto shm = world.split_shared();

  const int array_size = 23;

  // Only rank 0 allocates the shared array
  mpi::shared_window<int> win{shm, shm.rank() == 0 ? array_size : 0};
  std::span array_view{win.base(0), static_cast<std::size_t>(win.size(0))};

  // Fill array in parallel: each rank fills its chunk with array indices
  win.fence();
  for (auto i : mpi::chunk(itertools::range(array_size), shm)) { array_view[i] = static_cast<int>(i); }
  win.fence();

  // Total sum is just sum of numbers in interval [0, array_size)
  int sum = std::accumulate(array_view.begin(), array_view.end(), int{0});
  EXPECT_EQ(sum, (array_size * (array_size - 1)) / 2);
}

TEST(MPI, WindowDistributedSharedArray) {
  mpi::communicator world;
  auto island_comm = world.split_shared();

  // Number of total array elements (prime number to make it a bit more exciting)
  const int array_size_total = 197;

  // Create communicator of island leaders (rank 0 on each node)
  bool is_head   = island_comm.rank() == 0;
  auto head_comm = world.split(is_head ? 0 : MPI_UNDEFINED);

  // Each world rank gets a chunk of the global array
  auto [my_start, my_end] = itertools::chunk_range(0, array_size_total, world.size(), world.rank());
  int my_chunk_size       = static_cast<int>(my_end - my_start);

  // Gather all chunk sizes within the island
  auto island_chunk_sizes = mpi::all_gather(my_chunk_size, island_comm);
  int island_array_size   = std::accumulate(island_chunk_sizes.begin(), island_chunk_sizes.end(), int{0});

  // Allocate shared array combining all island ranks' chunks
  mpi::shared_window<int> win{island_comm, is_head ? island_array_size : 0};
  std::span array_view(win.base(0), island_array_size);

  // Calculate offset within the island's shared array
  int my_offset = std::accumulate(island_chunk_sizes.begin(), island_chunk_sizes.begin() + island_comm.rank(), int{0});

  // Each rank fills its chunk with global indices
  win.fence();
  auto my_chunk = array_view.subspan(my_offset, my_chunk_size);
  for (int i = 0; i < my_chunk_size; ++i) { my_chunk[i] = static_cast<int>(my_start + i); }
  win.fence();

  // Partial sum over my chunk
  int my_sum = std::accumulate(my_chunk.begin(), my_chunk.end(), int{0});

  // Partial sum over each island
  int island_sum = mpi::reduce(my_sum, island_comm);

  // Calculate Total sum on head ranks
  int total_sum = 0;
  if (is_head) { total_sum = mpi::reduce(island_sum, head_comm); }
  mpi::broadcast(total_sum, world);

  // Total sum is just sum of numbers in interval [0, array_size_total)
  EXPECT_EQ(total_sum, (array_size_total * (array_size_total - 1)) / 2);
}

MPI_TEST_MAIN;
