/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Ioan Sucan */

#include <custom/ompl_interface/detail/constrained_valid_state_sampler.h>
#include <custom/ompl_interface/model_based_planning_context.h>

#include <utility>

namespace custom_ompl_interface
{
static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit.custom_ompl_planning.constrained_valid_state_sampler");
}  // namespace custom_ompl_interface

custom_ompl_interface::ValidConstrainedSampler::ValidConstrainedSampler(const ModelBasedPlanningContext* pc,
                                                                 kinematic_constraints::KinematicConstraintSetPtr ks,
                                                                 constraint_samplers::ConstraintSamplerPtr cs)
  : ob::ValidStateSampler(pc->getOMPLSimpleSetup()->getSpaceInformation().get())
  , planning_context_(pc)
  , kinematic_constraint_set_(std::move(ks))
  , constraint_sampler_(std::move(cs))
  , work_state_(pc->getCompleteInitialRobotState())
{
  if (!constraint_sampler_)
    default_sampler_ = si_->allocStateSampler();
  inv_dim_ = si_->getStateSpace()->getDimension() > 0 ? 1.0 / (double)si_->getStateSpace()->getDimension() : 1.0;
  RCLCPP_DEBUG(LOGGER, "Constructed a ValidConstrainedSampler instance at address %p", this);
}

bool custom_ompl_interface::ValidConstrainedSampler::project(ompl::base::State* state)
{
  if (constraint_sampler_)
  {
    planning_context_->getOMPLStateSpace()->copyToRobotState(work_state_, state);
    if (constraint_sampler_->project(work_state_, planning_context_->getMaximumStateSamplingAttempts()))
    {
      if (kinematic_constraint_set_->decide(work_state_).satisfied)
      {
        planning_context_->getOMPLStateSpace()->copyToOMPLState(state, work_state_);
        return true;
      }
    }
  }
  return false;
}

bool custom_ompl_interface::ValidConstrainedSampler::sample(ob::State* state)
{
  if (constraint_sampler_)
  {
    if (constraint_sampler_->sample(work_state_, planning_context_->getCompleteInitialRobotState(),
                                    planning_context_->getMaximumStateSamplingAttempts()))
    {
      if (kinematic_constraint_set_->decide(work_state_).satisfied)
      {
        planning_context_->getOMPLStateSpace()->copyToOMPLState(state, work_state_);
        return true;
      }
    }
  }
  else
  {
    default_sampler_->sampleUniform(state);
    planning_context_->getOMPLStateSpace()->copyToRobotState(work_state_, state);
    if (kinematic_constraint_set_->decide(work_state_).satisfied)
      return true;
  }

  return false;
}

bool custom_ompl_interface::ValidConstrainedSampler::sampleNear(ompl::base::State* state, const ompl::base::State* near,
                                                         const double distance)
{
  if (!sample(state))
    return false;
  double total_d = si_->distance(state, near);
  if (total_d > distance)
  {
    double dist = pow(rng_.uniform01(), inv_dim_) * distance;
    si_->getStateSpace()->interpolate(near, state, dist / total_d, state);
    planning_context_->getOMPLStateSpace()->copyToRobotState(work_state_, state);
    if (!kinematic_constraint_set_->decide(work_state_).satisfied)
      return false;
  }
  return true;
}
