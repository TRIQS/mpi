.. highlight:: bash

.. _install:

Install mpi
*******************

Compiling mpi from source
===============================

.. note:: To guarantee reproducibility in scientific calculations we strongly recommend the use of a stable `release <https://github.com/TRIQS/mpi/releases>`_ of mpi.

Installation steps
------------------

#. Download the source code of the latest stable version by cloning the ``TRIQS/mpi`` repository from GitHub::

     $ git clone https://github.com/TRIQS/mpi mpi.src

#. Create and move to a new directory where you will compile the code::

     $ mkdir mpi.build && cd mpi.build

#. In the build directory call cmake, including any additional custom CMake options, see below::

     $ cmake -DCMAKE_INSTALL_PREFIX=path_to_install_dir ../mpi.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Versions
--------

To use a particular version, go into the directory with the sources, and look at all available versions::

     $ cd mpi.src && git tag

Checkout the version of the code that you want::

     $ git checkout 2.1.0

and follow steps 2 to 4 above to compile the code.

Custom CMake options
--------------------

The compilation of ``mpi`` can be configured using CMake-options::

    cmake ../mpi.src -DOPTION1=value1 -DOPTION2=value2 ...

+-----------------------------------------+-----------------------------------------------+
| Options                                 | Syntax                                        |
+=========================================+===============================================+
| Specify an installation path            | -DCMAKE_INSTALL_PREFIX=path_to_mpi      |
+-----------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                 | -DCMAKE_BUILD_TYPE=Debug                      |
+-----------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)       | -DBuild_Tests=OFF                             |
+-----------------------------------------+-----------------------------------------------+
| Build the documentation                 | -DBuild_Documentation=ON                      |
+-----------------------------------------+-----------------------------------------------+
