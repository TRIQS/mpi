#include "mpi/mpi.hpp"
#include "mpi/monitor.hpp"
#include <vector>
#include <gtest/gtest.h>
#include <unistd.h>

const int delta_tau_sleep = 1000; // in micro second : 3 ms

// fastest_node : position of the fastest node
bool test(mpi::communicator c, int fastest_node, int rank_failing, int iteration_failure = 3) {

  const int N     = 10;
  const long size = c.size();
  int sleeptime   = delta_tau_sleep * (((c.rank() - fastest_node + size) % size) + 1);
  std::cerr << "Node " << c.rank() << ": sleeptime " << sleeptime << std::endl;

  mpi::monitor M{c};

  for (int i = 0; (!M.should_stop()) and (i < N); ++i) {
    usleep(sleeptime);

    std::cerr << "N=" << c.rank() << " i=" << i << std::endl;

    if ((c.rank() == rank_failing) and (i >= iteration_failure)) {
      std::cerr << "Node " << c.rank() << " is failing" << std::endl;
      M.request_emergency_stop();
      M.request_emergency_stop(); // 2nd call should not do anything
    }
    if (i == N - 1) { std::cerr << "Node " << c.rank() << " done all tasks" << std::endl; }
  }

  bool success = M.finalize();
  std::cerr << "Ending on node " << c.rank() << std::endl;
  return success;
}

// ------------------------

TEST(MPI_Monitor, NoFailure) {
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_TRUE(test(world, i, -1));
    world.barrier();
  }
}

// ------------------------

TEST(MPI_Monitor, WithFailure) {
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    bool has_failure = (world.size() > 1 ? false : true); // No failure if only rank 0 exists
    EXPECT_EQ(test(world, i, 1), has_failure);
    world.barrier();
  }
}

// ------------------------

TEST(MPI_Monitor, WithFailureOnRoot) {
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_EQ(test(world, i, 0), false);
    world.barrier();
  }
}

MPI_TEST_MAIN;
