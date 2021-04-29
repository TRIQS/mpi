#include "mpi/mpi.hpp"
#include <gtest/gtest.h>
#include <unistd.h>

using namespace mpi;

TEST(MPI, barrier) {

  communicator world{};

  if (world.rank() == 0) {
    // Pretend to do something only in rank 0
    sleep(1);
  }
  // synchronize all ranks calling the normal MPI_barrier function by providing 0 as arg
  world.barrier(0);
  // now do the same with the poll barrier which reduces CPU load
  // to clearly the difference in htop one can simply set sleep(10)
  if (world.rank() == 0) {
    // Pretend to do something only in rank 0
    sleep(1);
  }
  // synchronize all ranks each 0.1 sec
  world.barrier(100);
  if (world.rank() == 0) {
    // Pretend to do something only in rank 0
    sleep(1);
  }
  // synchronize all ranks each 0.1 sec
  world.barrier(100);
}

MPI_TEST_MAIN;
