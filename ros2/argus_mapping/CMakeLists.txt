cmake_minimum_required(VERSION 3.16)
project(argus_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros_com REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EsdfGrid.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Pull in argus as a subdirectory to get the `argus_esdf` target — same
# consumption pattern argus_px4_bridge already uses for `quadrotor_mpc`.
# EXCLUDE_FROM_ALL: this package only needs argus_esdf, not the acados-linked
# argus_mpc demo executable that argus's own CMakeLists.txt also defines —
# without it, `colcon build --packages-select argus_mapping` would still
# build that unrelated target on every run.
set(ARGUS_SOURCE_DIR "$ENV{HOME}/thesis_to_cpp_ws/argus" CACHE PATH "Path to the argus repo")
add_subdirectory(${ARGUS_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/argus EXCLUDE_FROM_ALL)

add_executable(argus_mapping_node src/argus_mapping_node.cpp)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(argus_mapping_node
  argus_esdf
  px4_ros_com::frame_transforms
  "${cpp_typesupport_target}"
)

ament_target_dependencies(argus_mapping_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  px4_msgs
  px4_ros_com
  Eigen3
)

install(TARGETS argus_mapping_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
