#include "mpi/mpi.hpp"
#include "mpi/monitor.hpp"
#include <vector>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>

const int delta_tau_sleep = 30; // in ms

// fastest_node : position of the fastest node
bool test(mpi::communicator c, int fastest_node, int number_node_failing, int iteration_failure = 3) {

  const int N     = 10;
  const long size = c.size();
  int sleeptime   = delta_tau_sleep * (((c.rank() - fastest_node + size) % size) + 1);
  std::cerr << "nde " << c.rank() << "sleeptime " << sleeptime << std::endl;

  mpi::monitor M{c};

  for (int i = 0; (!M.should_stop()) and (i < N); ++i) {

    std::this_thread::sleep_for(std::chrono::milliseconds(sleeptime));

    std::cerr << "N" << c.rank() << " i = " << i << std::endl;

    if ((c.rank() == number_node_failing) and (i >= iteration_failure)) {
      std::cerr << "node : " << c.rank() << " is failing" << std::endl;
      M.request_emergency_stop();
    }
  }
  bool success = M.success();
  std::cerr << "Ending on node " << c.rank() << std::endl;
  return success;
}

// ------------------------

TEST(MPI_MONITOR, NoFailure) {
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "Node " << i << "is the fastest" << std::endl;
    EXPECT_TRUE(test(world, i, -1));
    world.barrier();
  }
}

// ------------------------

TEST(MPI_MONITOR, WithFailure) {
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "Node " << i << "is the fastest" << std::endl;
    EXPECT_EQ(test(world, i, 1), (world.size() > 1 ? false : true));
    world.barrier();
  }
}

MPI_TEST_MAIN;
