cmake_minimum_required(VERSION 3.8)
project(my_package)

find_package(ament_cmake REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  my_package/observer_node.py
  my_package/joystick_node.py
  my_package/guidance_node.py
  my_package/observer_node.py
  my_package/thrust_allocation_node.py
  my_package/controller_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  my_package/helpers
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
