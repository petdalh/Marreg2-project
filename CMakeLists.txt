cmake_minimum_required(VERSION 3.8)
project(my_package)

find_package(ament_cmake REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  my_package/my_node.py
  my_package/my_publisher.py
  my_package/joystick_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
