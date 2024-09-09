// Copyright (c) 2020-2024 Simons Foundation
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
// Authors: Philipp Dumitrescu, Thomas Hahn, Olivier Parcollet

#include <gtest/gtest.h>
#include <mpi/monitor.hpp>
#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <unistd.h>

// in micro second = 1 milli second
const int delta_tau_sleep = 1000;

// Monitor all nodes while some of them might report an event.
//
// c: MPI communicator.
// fastest_node: Rank of the fastest node.
// rank_reporting: Ranks of the nodes that will report an event.
// all_events: If true, the all_events_occurred() function will be used instead of some_event_occurred().
// iteration_event: Iteration at which the nodes will report an event.
bool test_monitor(mpi::communicator c, int fastest_node, std::vector<int> rank_reporting, bool all_events = false, int iteration_event = 3) {
  const int niter = 10;
  const int size  = c.size();
  int sleeptime   = delta_tau_sleep * (((c.rank() - fastest_node + size) % size) + 1);
  bool will_fail  = std::any_of(rank_reporting.cbegin(), rank_reporting.cend(), [&c](int i) { return i == c.rank(); });
  std::cerr << "Node " << c.rank() << ": sleeptime " << sleeptime << std::endl;

  mpi::monitor monitor{c};
  auto events_occurred = [all_events, &monitor]() { return all_events ? monitor.all_events_occurred() : monitor.some_event_occurred(); };

  for (int i = 0; (!events_occurred()) and (i < niter); ++i) {
    usleep(sleeptime);
    std::cerr << "Node " << c.rank() << " is in iteration " << i << std::endl;
    if (will_fail and (i >= iteration_event)) {
      std::cerr << "Node " << c.rank() << " is failing" << std::endl;
      monitor.report_local_event();
      monitor.report_local_event(); // 2nd call should not resend MPI message
    }
    if (i == niter - 1) { std::cerr << "Node " << c.rank() << " has done all tasks" << std::endl; }
  }

  monitor.finalize_communications();
  std::cerr << "Ending on node " << c.rank() << std::endl;
  return not events_occurred();
}

TEST(MPI, MonitorNoEvent) {
  // no event
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_TRUE(test_monitor(world, i, {}));
    world.barrier();
    EXPECT_TRUE(test_monitor(world, i, {}, true));
    world.barrier();
  }
}

TEST(MPI, MonitorOneEventOnRoot) {
  // one event on root node
  usleep(1000);
  mpi::communicator world;
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_EQ(test_monitor(world, i, {0}), false);
    world.barrier();
    EXPECT_EQ(test_monitor(world, i, {0}, true), world.size() > 1);
    world.barrier();
  }
  usleep(1000);
}

TEST(MPI, MonitorOneEventOnNonRoot) {
  // one event on non-root node
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 2) {
    if (world.rank() == 0) std::cerr << "This test is repeating previous tests if world.size() < 2. Skipping!" << std::endl;
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      bool has_failure = (world.size() > 1 ? false : true); // No failure if only rank 0 exists
      EXPECT_EQ(test_monitor(world, i, {1}), has_failure);
      world.barrier();
      EXPECT_EQ(test_monitor(world, i, {1}, true), world.size() > 1);
      world.barrier();
    }
  }
  usleep(1000);
}

TEST(MPI, MonitorTwoEventsWithRoot) {
  // two events on nodes including the root process
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 2) {
    if (world.rank() == 0) std::cerr << "This test is repeating previous tests if world.size() < 2. Skipping!" << std::endl;
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      EXPECT_EQ(test_monitor(world, i, {0, 1}), false);
      world.barrier();
      EXPECT_EQ(test_monitor(world, i, {0, 1}, true), world.size() > 2);
      world.barrier();
    }
  }
  usleep(1000);
}

TEST(MPI, MonitorTwoEventsWithoutRoot) {
  // two events on nodes excluding the root process
  usleep(1000);
  mpi::communicator world;
  if (world.size() < 3) {
    if (world.rank() == 0) { std::cerr << "This test is repeating previous tests if world.size() < 3. Skipping!" << std::endl; }
  } else {
    for (int i = 0; i < world.size(); ++i) {
      world.barrier();
      if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
      EXPECT_EQ(test_monitor(world, i, {1, 2}), false);
      world.barrier();
      EXPECT_EQ(test_monitor(world, i, {1, 2}, true), world.size() > 2);
      world.barrier();
    }
  }
  usleep(1000);
}

TEST(MPI, MonitorAllEvents) {
  // events on all nodes
  usleep(1000);
  mpi::communicator world;
  std::vector<int> rank_reporting(world.size());
  std::iota(rank_reporting.begin(), rank_reporting.end(), 0);
  for (int i = 0; i < world.size(); ++i) {
    world.barrier();
    if (world.rank() == 0) std::cerr << "***\nNode " << i << " is the fastest" << std::endl;
    EXPECT_FALSE(test_monitor(world, i, rank_reporting));
    world.barrier();
    EXPECT_FALSE(test_monitor(world, i, rank_reporting, true));
    world.barrier();
  }
  usleep(1000);
}

TEST(MPI, MultipleMonitors) {
  // test multiple monitors
  usleep(1000);
  mpi::communicator world;
  auto dup = world.duplicate();
  auto dupdup = world.duplicate();
  mpi::monitor monitor1{world};
  mpi::monitor monitor2{dup};
  mpi::monitor monitor3{dupdup};
  if (world.rank() == 0) {
    monitor3.report_local_event();
  }
  monitor2.report_local_event();
  monitor1.finalize_communications();
  monitor2.finalize_communications();
  monitor3.finalize_communications();
  EXPECT_FALSE(monitor1.some_event_occurred());
  EXPECT_FALSE(monitor1.all_events_occurred());
  EXPECT_TRUE(monitor2.some_event_occurred());
  EXPECT_TRUE(monitor2.all_events_occurred());
  EXPECT_TRUE(monitor3.some_event_occurred());
  if (world.size() == 1) EXPECT_TRUE(monitor3.all_events_occurred());
  else EXPECT_FALSE(monitor3.all_events_occurred());
  dup.free();
  dupdup.free();
  usleep(1000);
}

MPI_TEST_MAIN;
