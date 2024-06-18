set(MOVEIT_LIB_NAME custom_ompl_interface)

add_library(${MOVEIT_LIB_NAME} SHARED
  src/ompl_interface.cpp
  src/planning_context_manager.cpp
  src/model_based_planning_context.cpp
  src/parameterization/model_based_state_space.cpp
  src/parameterization/model_based_state_space_factory.cpp
  src/parameterization/joint_space/constrained_planning_state_space.cpp
  src/parameterization/joint_space/constrained_planning_state_space_factory.cpp
  src/parameterization/joint_space/joint_model_state_space.cpp
  src/parameterization/joint_space/joint_model_state_space_factory.cpp
  src/parameterization/work_space/pose_model_state_space.cpp
  src/parameterization/work_space/pose_model_state_space_factory.cpp
  src/detail/ompl_constraints.cpp
  src/detail/threadsafe_state_storage.cpp
  src/detail/state_validity_checker.cpp
  src/detail/projection_evaluators.cpp
  src/detail/goal_union.cpp
  src/detail/constraints_library.cpp
  src/detail/constrained_sampler.cpp
  src/detail/constrained_valid_state_sampler.cpp
  src/detail/constrained_goal_sampler.cpp
)
set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES VERSION "${${PROJECT_NAME}_VERSION}")

find_package(OpenMP REQUIRED)

# Used to link in ODE, an OMPL dependency, on macOS
if(APPLE)
  target_link_directories(${MOVEIT_LIB_NAME} PUBLIC ${OMPL_LIBRARY_DIRS})
endif()

ament_target_dependencies(${MOVEIT_LIB_NAME}
  moveit_core
  moveit_msgs
  moveit_ros_planning
  rclcpp
  pluginlib
  tf2_eigen
  tf2_ros
  OMPL
  Boost
)
set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

add_library(custom_ompl_planner_plugin SHARED src/ompl_planner_manager.cpp)
set_target_properties(custom_ompl_planner_plugin PROPERTIES VERSION "${${PROJECT_NAME}_VERSION}")
ament_target_dependencies(custom_ompl_planner_plugin
  moveit_core
  moveit_ros_planning
  rclcpp
  pluginlib
  tf2_ros
  OMPL
  Boost
)
target_link_libraries(custom_ompl_planner_plugin ${MOVEIT_LIB_NAME})

install(DIRECTORY include/ DESTINATION include)
