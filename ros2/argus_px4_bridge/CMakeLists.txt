cmake_minimum_required(VERSION 3.16)
project(argus_px4_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros_com REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Pull in argus as a subdirectory to get the `quadrotor_mpc` target — same
# consumption pattern the argus repo's own CMakeLists.txt is built for
# (acados stays isolated inside argus; this package never touches it).
set(ARGUS_SOURCE_DIR "$ENV{HOME}/thesis_to_cpp_ws/argus" CACHE PATH "Path to the argus repo")
add_subdirectory(${ARGUS_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/argus)

add_executable(argus_bridge_node src/argus_bridge_node.cpp)

target_link_libraries(argus_bridge_node
  quadrotor_mpc
  px4_ros_com::frame_transforms
)

ament_target_dependencies(argus_bridge_node
  rclcpp
  px4_msgs
  px4_ros_com
  Eigen3
  geometry_msgs
  nav_msgs
  visualization_msgs
)

install(TARGETS argus_bridge_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME})

ament_package()
