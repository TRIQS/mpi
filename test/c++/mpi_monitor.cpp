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
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#include <gtest/gtest.h>
#include <mpi/monitor.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <unistd.h>

// in micro second = 1 milli second
const int delta_tau_sleep = 1000;

// Monitor all nodes while some of them might fail.
//
// c: MPI communicator
// fastest_node: rank of the fastest node
// rank_failing: ranks of the nodes that will fail
// iteration_failure: iteration at which the nodes will fail
bool test(mpi::communicator c, int fastest_node, std::vector<int> rank_failing, int iteration_failure = 3) {
  const int niter = 10;
  const int size  = c.size();
  int sleeptime   = delta_tau_sleep * (((c.rank() - fastest_node + size) % size) + 1);
  bool will_fail  = std::any_of(rank_failing.cbegin(), rank_failing.cend(), [&c](int i) { return i == c.rank(); });
  std::cerr << "Node " << c.rank() << ": sleeptime " << sleeptime << std::endl;

  mpi::monitor monitor{c};

  for (int i = 0; (!monitor.emergency_occured()) and (i < niter); ++i) {
    usleep(sleeptime);
    std::cerr << "Node " << c.rank() << "is in iteration " << i << std::endl;
    if (will_fail and (i >= iteration_failure)) {
      std::cerr << "Node " << c.rank() << " is failing" << std::endl;
      monitor.request_emergency_stop();
      monitor.request_emergency_stop(); // 2nd call should not resend MPI message
    }
    if (i == niter - 1) { std::cerr << "Node " << c.rank() << " has done all tasks" << std::endl; }
  }

  monitor.finalize_communications();
  std::cerr << "Ending on node " << c.rank() << std::endl;
  return not monitor.emergency_occured();
}

TEST(MPI, MonitorNoFailure) {
  // no failure
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_TRUE(test(world, i, {}));
    world.barrier();
  }
}

TEST(MPI, MonitorOneFailureOnRoot) {
  // root node fails
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

TEST(MPI, MonitorOneFailureOnNonRoot) {
  // one non-root node fails
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

TEST(MPI, MonitorTwoFailuresWithRoot) {
  // two nodes fail including the root process
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

TEST(MPI, MonitorTwoFailuresWithoutRoot) {
  // two nodes fail excluding the root process
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
