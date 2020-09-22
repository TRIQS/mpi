#include "mpi/mpi.hpp"
#include "mpi/monitor.hpp"
#include <vector>
#include <gtest/gtest.h>
#include <unistd.h>

const int delta_tau_sleep = 1000; // in micro second : 3 ms

// fastest_node : position of the fastest node
bool test(mpi::communicator c, int fastest_node, std::vector<int> rank_failing, int iteration_failure = 3) {

  const int N     = 10;
  const long size = c.size();
  int sleeptime   = delta_tau_sleep * (((c.rank() - fastest_node + size) % size) + 1);
  bool will_fail  = std::any_of(rank_failing.cbegin(), rank_failing.cend(), [&c](int i) { return i == c.rank(); });
  std::cerr << "Node " << c.rank() << ": sleeptime " << sleeptime << std::endl;

  mpi::monitor M{c};

  for (int i = 0; (!M.emergency_occured()) and (i < N); ++i) {
    usleep(sleeptime);

    std::cerr << "N=" << c.rank() << " i=" << i << std::endl;

    if (will_fail and (i >= iteration_failure)) {
      std::cerr << "Node " << c.rank() << " is failing" << std::endl;
      M.request_emergency_stop();
      M.request_emergency_stop(); // 2nd call should not resend MPI message
    }
    if (i == N - 1) { std::cerr << "Node " << c.rank() << " done all tasks" << std::endl; }
  }

  M.finalize_communications();
  std::cerr << "Ending on node " << c.rank() << std::endl;
  return not M.emergency_occured();
}

// ------------------------

TEST(MPI_Monitor, NoFailure) {
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_TRUE(test(world, i, {}));
    world.barrier();
  }
}

// ------------------------

TEST(MPI_Monitor, OneFailureOnRoot) {
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_EQ(test(world, i, {0}), false);
    world.barrier();
  }
  usleep(1000);
}

TEST(MPI_Monitor, OneFailureNoRoot) {
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 2) {
    if (world.rank() == 0) std::cerr << "This test is repeating previous tests if world.size() < 2. Skipping!" << std::endl;
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      bool has_failure = (world.size() > 1 ? false : true); // No failure if only rank 0 exists
      EXPECT_EQ(test(world, i, {1}), has_failure);
      world.barrier();
    }
  }
  usleep(1000);
}

TEST(MPI_Monitor, TwoFailuresWithRoot) {
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 2) {
    if (world.rank() == 0) std::cerr << "This test is repeating previous tests if world.size() < 2. Skipping!" << std::endl;
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      EXPECT_EQ(test(world, i, {0, 1}), false);
      world.barrier();
    }
  }
  usleep(1000);
}

TEST(MPI_Monitor, TwoFailuresWithoutRoot) {
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 3) {
    if (world.rank() == 0) { std::cerr << "This test is repeating previous tests if world.size() < 3. Skipping!" << std::endl; }
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      EXPECT_EQ(test(world, i, {1, 2}), false);
      world.barrier();
    }
  }
  usleep(1000);
}

MPI_TEST_MAIN;
