@page ex2 Example 2: Use monitor to communicate errors

[TOC]

In this example, we show how to use the mpi::monitor class to communicate and process errors across a communicator.

```cpp
#include <mpi/mpi.hpp>
#include <mpi/monitor.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // initialize monitor
  mpi::monitor monitor(world);

  // in case a stop has been requested, print some info and return true
  auto stop = [&monitor, world](int i) {
    bool res = false;
    if (monitor.emergency_occured()) {
      std::cerr << "Processor " << world.rank() << ": After " << i << " steps an emergency stop has been received.\n";
      res = true;
    }
    return res;
  };

  // loop as long as no stop has been requested
  int rank_to_req = 3;
  for (int i = 0; i < 1000000; ++i) {
    // request a stop on processor 3
    if (world.rank() == rank_to_req) {
      std::cerr << "Processor " << rank_to_req << ": Emergency stop requested.\n";
      monitor.request_emergency_stop();
    }

    // should we stop the loop?
    if (stop(i)) break;
  }

  // check if all processes finished without an error
  if (world.rank() == 0) {
    if (monitor.emergency_occured()) {
      std::cout << "Oh no! An error occurred somewhere.\n";
    } else {
      std::cout << "No worries, all processes finished without an error.\n";
    }
  }
}
```

Output (running with `-n 12`):

```
Processor 3: Emergency stop requested.
Processor 3: After 0 steps an emergency stop has been received.
Processor 2: After 5950 steps an emergency stop has been received.
Processor 4: After 10475 steps an emergency stop has been received.
Processor 5: After 7379 steps an emergency stop has been received.
Processor 6: After 8366 steps an emergency stop has been received.
Processor 7: After 1302 steps an emergency stop has been received.
Processor 8: After 1155 steps an emergency stop has been received.
Processor 9: After 14445 steps an emergency stop has been received.
Processor 11: After 9287 steps an emergency stop has been received.
Processor 0: After 0 steps an emergency stop has been received.
Processor 1: After 7443 steps an emergency stop has been received.
Processor 10: After 1321 steps an emergency stop has been received.
Oh no! An error occurred somewhere.
```

Output (running with `-n 3`):

```
No worries, all processes finished without an error.
```