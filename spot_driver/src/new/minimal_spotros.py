"""
Minimal yet complete Spot ROS driver in python based on the following principles:
* Easy to read, simple, minimal Spot ROS driver
* Up-to-date with Spot SDK v4
* No external dependencies
* No clutter, unnecessary code
* Verbose naming of variables and functions
* Reduced call stack: removing unnecessary function chaining
"""
# Standard
import time
import math
import os
import rospy
import numpy as np
import functools
import actionlib
import logging
import threading
import bosdyn.geometry
import tf2_ros

# ROS
from spot_msgs.msg import EStopState, EStopStateArray
from spot_msgs.msg import PowerState
from spot_msgs.srv import ListGraph, ListGraphResponse
from spot_msgs.srv import ClearBehaviorFault, ClearBehaviorFaultResponse
from spot_msgs.srv import SetLocomotion, SetLocomotionResponse
from spot_msgs.srv import SetVelocity, SetVelocityResponse
from spot_msgs.msg import TrajectoryAction, TrajectoryResult, TrajectoryFeedback
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistWithCovarianceStamped, Twist, Pose
from nav_msgs.msg import Odometry
from spot_msgs.msg import Metrics
from spot_msgs.msg import LeaseArray, LeaseResource
from spot_msgs.msg import FootState, FootStateArray
from spot_msgs.msg import EStopState, EStopStateArray
from spot_msgs.msg import WiFiState
from spot_msgs.msg import BehaviorFault, BehaviorFaultState
from spot_msgs.msg import SystemFault, SystemFaultState
from spot_msgs.msg import BatteryState, BatteryStateArray
from spot_msgs.msg import Feedback
from spot_msgs.msg import MobilityParams
from spot_msgs.msg import NavigateToAction, NavigateToResult, NavigateToFeedback

# BD Spot SDK
from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.power import safe_power_off, PowerClient, power_on
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
import bosdyn.api.robot_state_pb2 as robot_state_proto
from bosdyn.api import image_pb2
from bosdyn.api.geometry_pb2 import Quaternion, SE2VelocityLimit
from bosdyn.client import create_standard_sdk, ResponseError, RpcError
from bosdyn.client import math_helpers
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api import geometry_pb2, trajectory_pb2
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client import frame_helpers
from bosdyn.client.frame_helpers import get_odom_tform_body, get_vision_tform_body
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.api.graph_nav import nav_pb2
from bosdyn.api.graph_nav import map_pb2
from geometry_msgs.msg import PoseWithCovariance

from google.protobuf.timestamp_pb2 import Timestamp
from bosdyn.client.math_helpers import SE3Pose
from geometry_msgs.msg import TwistWithCovariance

friendly_joint_names = {}
"""Dictionary for mapping BD joint names to more friendly names"""
friendly_joint_names["fl.hx"] = "front_left_hip_x"
friendly_joint_names["fl.hy"] = "front_left_hip_y"
friendly_joint_names["fl.kn"] = "front_left_knee"
friendly_joint_names["fr.hx"] = "front_right_hip_x"
friendly_joint_names["fr.hy"] = "front_right_hip_y"
friendly_joint_names["fr.kn"] = "front_right_knee"
friendly_joint_names["hl.hx"] = "rear_left_hip_x"
friendly_joint_names["hl.hy"] = "rear_left_hip_y"
friendly_joint_names["hl.kn"] = "rear_left_knee"
friendly_joint_names["hr.hx"] = "rear_right_hip_x"
friendly_joint_names["hr.hy"] = "rear_right_hip_y"
friendly_joint_names["hr.kn"] = "rear_right_knee"

front_image_sources = ['frontleft_fisheye_image', 'frontright_fisheye_image', 'frontleft_depth', 'frontright_depth']
"""List of image sources for front image periodic query"""
side_image_sources = ['left_fisheye_image', 'right_fisheye_image', 'left_depth', 'right_depth']
"""List of image sources for side image periodic query"""
rear_image_sources = ['back_fisheye_image', 'back_depth']
"""List of image sources for rear image periodic query"""

class AsyncRobotState(AsyncPeriodicQuery):
    """Class to get robot state at regular intervals.  get_robot_state_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncRobotState, self).__init__("robot-state", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_state_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncMetrics(AsyncPeriodicQuery):
    """Class to get robot metrics at regular intervals.  get_robot_metrics_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncMetrics, self).__init__("robot-metrics", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_metrics_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncLease(AsyncPeriodicQuery):
    """Class to get lease state at regular intervals.  list_leases_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncLease, self).__init__("lease", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.list_leases_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncImageService(AsyncPeriodicQuery):
    """Class to get images at regular intervals.  get_image_from_sources_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback, image_requests):
        super(AsyncImageService, self).__init__("robot_image_service", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback
        self._image_requests = image_requests

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_image_async(self._image_requests)
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncIdle(AsyncPeriodicQuery):
    """Class to check if the robot is moving, and if not, command a stand with the set mobility parameters

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            spot_wrapper: A handle to the wrapper library
    """
    def __init__(self, client, logger, rate, spot_wrapper):
        super(AsyncIdle, self).__init__("idle", client, logger,
                                           period_sec=1.0/rate)

        self._spot_wrapper = spot_wrapper

    def _start_query(self):
        if self._spot_wrapper._last_stand_command != None:
            try:
                response = self._client.robot_command_feedback(self._spot_wrapper._last_stand_command)
                self._spot_wrapper._is_sitting = False
                if (response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status ==
                        basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING):
                    self._spot_wrapper._is_standing = True
                    self._spot_wrapper._last_stand_command = None
                else:
                    self._spot_wrapper._is_standing = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_stand_command = None

        if self._spot_wrapper._last_sit_command != None:
            try:
                self._spot_wrapper._is_standing = False
                response = self._client.robot_command_feedback(self._spot_wrapper._last_sit_command)
                if (response.feedback.synchronized_feedback.mobility_command_feedback.sit_feedback.status ==
                        basic_command_pb2.SitCommand.Feedback.STATUS_IS_SITTING):
                    self._spot_wrapper._is_sitting = True
                    self._spot_wrapper._last_sit_command = None
                else:
                    self._spot_wrapper._is_sitting = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_sit_command = None

        is_moving = False

        if self._spot_wrapper._last_velocity_command_time != None:
            if time.time() < self._spot_wrapper._last_velocity_command_time:
                is_moving = True
            else:
                self._spot_wrapper._last_velocity_command_time = None

        if self._spot_wrapper._last_trajectory_command != None:
            try:
                response = self._client.robot_command_feedback(self._spot_wrapper._last_trajectory_command)
                status = response.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
                # STATUS_AT_GOAL always means that the robot reached the goal. If the trajectory command did not
                # request precise positioning, then STATUS_NEAR_GOAL also counts as reaching the goal
                if status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL or \
                    (status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL and
                     not self._spot_wrapper._last_trajectory_command_precise):
                    self._spot_wrapper._at_goal = True
                    # Clear the command once at the goal
                    self._spot_wrapper._last_trajectory_command = None
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_GOING_TO_GOAL:
                    is_moving = True
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL:
                    is_moving = True
                    self._spot_wrapper._near_goal = True
                else:
                    self._spot_wrapper._last_trajectory_command = None
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_trajectory_command = None

        self._spot_wrapper._is_moving = is_moving

        if self._spot_wrapper.is_standing and not self._spot_wrapper.is_moving:
            self._spot_wrapper.stand(False)

class SpotWrapper():
    """Generic wrapper class to encompass release 1.1.4 API features as well as maintaining leases automatically"""
    def __init__(self, username, password, hostname, logger, rates = {}, callbacks = {}):
        self._username = username
        self._password = password
        self._hostname = hostname
        self._logger = logger
        self._rates = rates
        self._callbacks = callbacks
        self._keep_alive = True
        self._valid = True

        self._mobility_params = RobotCommandBuilder.mobility_params()
        self._is_standing = False
        self._is_sitting = True
        self._is_moving = False
        self._at_goal = False
        self._near_goal = False
        self._last_stand_command = None
        self._last_sit_command = None
        self._last_trajectory_command = None
        self._last_trajectory_command_precise = None
        self._last_velocity_command_time = None

        self._front_image_requests = []
        for source in front_image_sources:
            self._front_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        self._side_image_requests = []
        for source in side_image_sources:
            self._side_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        self._rear_image_requests = []
        for source in rear_image_sources:
            self._rear_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        try:
            self._sdk = create_standard_sdk('ros_spot')
        except Exception as e:
            self._logger.error("Error creating SDK object: %s", e)
            self._valid = False
            return

        self._robot = self._sdk.create_robot(self._hostname)

        try:
            self._robot.authenticate(self._username, self._password)
            self._robot.start_time_sync()
        except RpcError as err:
            self._logger.error("Failed to communicate with robot: %s", err)
            self._valid = False
            return

        if self._robot:
            # Clients
            try:
                self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
                self._robot_command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
                self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)
                self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
                self._lease_wallet = self._lease_client.lease_wallet
                self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
            except Exception as e:
                self._logger.error("Unable to create client service: %s", e)
                self._valid = False
                return

            # Store the most recent knowledge of the state of the robot based on rpc calls.
            self._current_graph = None
            self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
            self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
            self._current_edge_snapshots = dict()  # maps id to edge snapshot
            self._current_annotation_name_to_wp_id = dict()

            # Async Tasks
            self._async_task_list = []
            self._robot_state_task = AsyncRobotState(self._robot_state_client, self._logger, max(0.0, self._rates.get("robot_state", 0.0)), self._callbacks.get("robot_state", lambda:None))
            self._robot_metrics_task = AsyncMetrics(self._robot_state_client, self._logger, max(0.0, self._rates.get("metrics", 0.0)), self._callbacks.get("metrics", lambda:None))
            self._lease_task = AsyncLease(self._lease_client, self._logger, max(0.0, self._rates.get("lease", 0.0)), self._callbacks.get("lease", lambda:None))
            self._front_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("front_image", 0.0)), self._callbacks.get("front_image", lambda:None), self._front_image_requests)
            self._side_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("side_image", 0.0)), self._callbacks.get("side_image", lambda:None), self._side_image_requests)
            self._rear_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("rear_image", 0.0)), self._callbacks.get("rear_image", lambda:None), self._rear_image_requests)
            self._idle_task = AsyncIdle(self._robot_command_client, self._logger, 10.0, self)

            self._async_tasks = AsyncTasks(
                [self._robot_state_task, self._robot_metrics_task, self._lease_task, self._front_image_task, self._side_image_task, self._rear_image_task, self._idle_task])

            self._robot_id = None
            self._lease = None

    @property
    def logger(self):
        """Return logger instance of the SpotWrapper"""
        return self._logger

    @property
    def is_valid(self):
        """Return boolean indicating if the wrapper initialized successfully"""
        return self._valid

    @property
    def id(self):
        """Return robot's ID"""
        return self._robot_id

    @property
    def robot_state(self):
        """Return latest proto from the _robot_state_task"""
        return self._robot_state_task.proto

    @property
    def metrics(self):
        """Return latest proto from the _robot_metrics_task"""
        return self._robot_metrics_task.proto

    @property
    def lease(self):
        """Return latest proto from the _lease_task"""
        return self._lease_task.proto

    @property
    def front_images(self):
        """Return latest proto from the _front_image_task"""
        return self._front_image_task.proto

    @property
    def side_images(self):
        """Return latest proto from the _side_image_task"""
        return self._side_image_task.proto

    @property
    def rear_images(self):
        """Return latest proto from the _rear_image_task"""
        return self._rear_image_task.proto

    @property
    def is_standing(self):
        """Return boolean of standing state"""
        return self._is_standing

    @property
    def is_sitting(self):
        """Return boolean of standing state"""
        return self._is_sitting

    @property
    def is_moving(self):
        """Return boolean of walking state"""
        return self._is_moving

    @property
    def near_goal(self):
        return self._near_goal

    @property
    def at_goal(self):
        return self._at_goal

    @property
    def time_skew(self):
        """Return the time skew between local and spot time"""
        return self._robot.time_sync.endpoint.clock_skew

    def robotToLocalTime(self, timestamp):
        """Takes a timestamp and an estimated skew and return seconds and nano seconds in local time

        Args:
            timestamp: google.protobuf.Timestamp
        Returns:
            google.protobuf.Timestamp
        """

        rtime = Timestamp()

        rtime.seconds = timestamp.seconds - self.time_skew.seconds
        rtime.nanos = timestamp.nanos - self.time_skew.nanos
        if rtime.nanos < 0:
            rtime.nanos = rtime.nanos + 1000000000
            rtime.seconds = rtime.seconds - 1

        # Workaround for timestamps being incomplete
        if rtime.seconds < 0:
            rtime.seconds = 0
            rtime.nanos = 0

        return rtime

    def _robot_command(self, command_proto, end_time_secs=None, timesync_endpoint=None):
        """Generic blocking function for sending commands to robots.

        Args:
            command_proto: robot_command_pb2 object to send to the robot.  Usually made with RobotCommandBuilder
            end_time_secs: (optional) Time-to-live for the command in seconds
            timesync_endpoint: (optional) Time sync endpoint
        """
        try:
            id = self._robot_command_client.robot_command(lease=None, command=command_proto, end_time_secs=end_time_secs, timesync_endpoint=timesync_endpoint)
            return True, "Success", id
        except Exception as e:
            return False, str(e), None

    def velocity_cmd(self, v_x, v_y, v_rot, cmd_duration=0.125):
        """Send a velocity motion command to the robot.

        Args:
            v_x: Velocity in the X direction in meters
            v_y: Velocity in the Y direction in meters
            v_rot: Angular velocity around the Z axis in radians
            cmd_duration: (optional) Time-to-live for the command in seconds.  Default is 125ms (assuming 10Hz command rate).
        """
        end_time=time.time() + cmd_duration
        response = self._robot_command(RobotCommandBuilder.synchro_velocity_command(
                                      v_x=v_x, v_y=v_y, v_rot=v_rot, params=self._mobility_params),
                                      end_time_secs=end_time, timesync_endpoint=self._robot.time_sync.endpoint)
        self._last_velocity_command_time = end_time
        return response[0], response[1]

    def trajectory_cmd(self, goal_x, goal_y, goal_heading, cmd_duration, frame_name='odom', precise_position=False):
        """Send a trajectory motion command to the robot.

        Args:
            goal_x: Position X coordinate in meters
            goal_y: Position Y coordinate in meters
            goal_heading: Pose heading in radians
            cmd_duration: Time-to-live for the command in seconds.
            frame_name: frame_name to be used to calc the target position. 'odom' or 'vision'
            precise_position: if set to false, the status STATUS_NEAR_GOAL and STATUS_AT_GOAL will be equivalent. If
            true, the robot must complete its final positioning before it will be considered to have successfully
            reached the goal.
        """
        self._at_goal = False
        self._near_goal = False
        self._last_trajectory_command_precise = precise_position
        self._logger.info("got command duration of {}".format(cmd_duration))
        end_time=time.time() + cmd_duration
        if frame_name == 'vision':
            vision_tform_body = frame_helpers.get_vision_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            vision_tform_goal = vision_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=vision_tform_goal.x,
                                goal_y=vision_tform_goal.y,
                                goal_heading=vision_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.VISION_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        elif frame_name == 'odom':
            odom_tform_body = frame_helpers.get_odom_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            odom_tform_goal = odom_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=odom_tform_goal.x,
                                goal_y=odom_tform_goal.y,
                                goal_heading=odom_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.ODOM_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        else:
            raise ValueError('frame_name must be \'vision\' or \'odom\'')
        if response[0]:
            self._last_trajectory_command = response[2]
        return response[0], response[1]


    def navigate_to(self, upload_path,
                    navigate_to,
                    initial_localization_fiducial=True,
                    initial_localization_waypoint=None):
        """ navigate with graph nav.

        Args:
           upload_path : Path to the root directory of the map.
           navigate_to : Waypont id string for where to goal
           initial_localization_fiducial : Tells the initializer whether to use fiducials
           initial_localization_waypoint : Waypoint id string of current robot position (optional)
        """
        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            upload_filepath = upload_path[:-1]
        else:
            upload_filepath = upload_path

        # Boolean indicating the robot's power state.
        # Check if the robot is powered on.
        self._started_powered_on = self.check_is_powered_on()


        # FIX ME somehow,,,, if the robot is stand, need to sit the robot before starting garph nav
        if self.is_standing and not self.is_moving:
            self.sit()

        # TODO verify estop  / claim / power_on
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        self._graph_nav_client.clear_graph(lease=self._lease.lease_proto)
        self._upload_graph_and_snapshots(upload_filepath)
        if initial_localization_fiducial:
            self._set_initial_localization_fiducial()
        if initial_localization_waypoint:
            self._set_initial_localization_waypoint([initial_localization_waypoint])
        self._list_graph_waypoint_and_edge_ids()
        self._get_localization_state()
        resp = self._navigate_to([navigate_to])

        return resp

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            self._logger.info("No waypoint provided as a destination for navigate to.")
            return

        self._lease = self._lease_wallet.get_lease()
        destination_waypoint = find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            self._logger.info("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keepalive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()

        # Navigate to the destination waypoint.
        is_finished = False
        nav_to_cmd_id = -1
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with  or killing the program).
            nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                               leases=[sublease.lease_proto])
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Update the lease and power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

        status = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            return True, "Successfully completed the navigation commands!"
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            return False, "Robot got lost when navigating the route, the robot will now sit down."
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            return False, "Robot got stuck when navigating the route, the robot will now sit down."
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            return False, "Robot is impaired."
        else:
            return False, "Navigation command is not complete yet."
 
    ## copy from spot-sdk/python/examples/graph_nav_command_line/graph_nav_command_line.py
    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        self._logger.info('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        self._logger.info('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            self._logger.error("No waypoint specified to initialize to.")
            return
        destination_waypoint = find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance = 0.2,
            max_yaw = 20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            self._logger.error("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = update_waypoints_and_edges(
            graph, localization_id, self._logger)
        return self._current_annotation_name_to_wp_id, self._current_edges


    def _upload_graph_and_snapshots(self, upload_filepath):
        """Upload the graph and snapshots to the robot."""
        self._logger.info("Loading the graph from disk into local storage...")
        with open(upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            self._logger.info("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            # Load the edge snapshots from disk.
            with open(upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        self._logger.info("Uploading the graph and snapshots to the robot...")
        self._graph_nav_client.upload_graph(lease=self._lease.lease_proto,
                                            graph=self._current_graph)
        # Upload the snapshots to the robot.
        for waypoint_snapshot in self._current_waypoint_snapshots.values():
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            self._logger.info("Uploaded {}".format(waypoint_snapshot.id))
        for edge_snapshot in self._current_edge_snapshots.values():
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            self._logger.info("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and it if is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            self._logger.info(
                   "Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have not status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self._logger.error("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self._logger.error("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self._logger.error("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

class DefaultCameraInfo(CameraInfo):
    """Blank class extending CameraInfo ROS topic that defaults most parameters"""
    def __init__(self):
        super().__init__()
        self.distortion_model = "plumb_bob"

        self.D.append(0)
        self.D.append(0)
        self.D.append(0)
        self.D.append(0)
        self.D.append(0)

        self.K[1] = 0
        self.K[3] = 0
        self.K[6] = 0
        self.K[7] = 0
        self.K[8] = 1

        self.R[0] = 1
        self.R[1] = 0
        self.R[2] = 0
        self.R[3] = 0
        self.R[4] = 1
        self.R[5] = 0
        self.R[6] = 0
        self.R[7] = 0
        self.R[8] = 1

        self.P[1] = 0
        self.P[3] = 0
        self.P[4] = 0
        self.P[7] = 0
        self.P[8] = 0
        self.P[9] = 0
        self.P[10] = 1
        self.P[11] = 0

def getImageMsg(data, spot_wrapper):
    """Takes the imag and  camera data and populates the necessary ROS messages

    Args:
        data: Image proto
        spot_wrapper: A SpotWrapper object
    Returns:
        (tuple):
            * Image: message of the image captured
            * CameraInfo: message to define the state and config of the camera that took the image
    """
    image_msg = Image()
    local_time = spot_wrapper.robotToLocalTime(data.shot.acquisition_time)
    image_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    image_msg.header.frame_id = data.shot.frame_name_image_sensor
    image_msg.height = data.shot.image.rows
    image_msg.width = data.shot.image.cols

    # Color/greyscale formats.
    # JPEG format
    if data.shot.image.format == image_pb2.Image.FORMAT_JPEG:
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = True
        image_msg.step = 3 * data.shot.image.cols
        image_msg.data = data.shot.image.data

    # Uncompressed.  Requires pixel_format.
    if data.shot.image.format == image_pb2.Image.FORMAT_RAW:
        # One byte per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            image_msg.encoding = "mono8"
            image_msg.is_bigendian = True
            image_msg.step = data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Three bytes per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            image_msg.encoding = "rgb8"
            image_msg.is_bigendian = True
            image_msg.step = 3 * data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Four bytes per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            image_msg.encoding = "rgba8"
            image_msg.is_bigendian = True
            image_msg.step = 4 * data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Little-endian uint16 z-distance from camera (mm).
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            image_msg.encoding = "16UC1"
            image_msg.is_bigendian = False
            image_msg.step = 2 * data.shot.image.cols
            image_msg.data = data.shot.image.data

    camera_info_msg = DefaultCameraInfo()
    local_time = spot_wrapper.robotToLocalTime(data.shot.acquisition_time)
    camera_info_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    camera_info_msg.header.frame_id = data.shot.frame_name_image_sensor
    camera_info_msg.height = data.shot.image.rows
    camera_info_msg.width = data.shot.image.cols

    camera_info_msg.K[0] = data.source.pinhole.intrinsics.focal_length.x
    camera_info_msg.K[2] = data.source.pinhole.intrinsics.principal_point.x
    camera_info_msg.K[4] = data.source.pinhole.intrinsics.focal_length.y
    camera_info_msg.K[5] = data.source.pinhole.intrinsics.principal_point.y

    camera_info_msg.P[0] = data.source.pinhole.intrinsics.focal_length.x
    camera_info_msg.P[2] = data.source.pinhole.intrinsics.principal_point.x
    camera_info_msg.P[5] = data.source.pinhole.intrinsics.focal_length.y
    camera_info_msg.P[6] = data.source.pinhole.intrinsics.principal_point.y

    return image_msg, camera_info_msg

def populateTransformStamped(time, parent_frame, child_frame, transform):
    """Populates a TransformStamped message

    Args:
        time: The time of the transform
        parent_frame: The parent frame of the transform
        child_frame: The child_frame_id of the transform
        transform: A transform to copy into a StampedTransform object. Should have position (x,y,z) and rotation (x,
        y,z,w) members
    Returns:
        TransformStamped message
    """
    new_tf = TransformStamped()
    new_tf.header.stamp = time
    new_tf.header.frame_id = parent_frame
    new_tf.child_frame_id = child_frame
    new_tf.transform.translation.x = transform.position.x
    new_tf.transform.translation.y = transform.position.y
    new_tf.transform.translation.z = transform.position.z
    new_tf.transform.rotation.x = transform.rotation.x
    new_tf.transform.rotation.y = transform.rotation.y
    new_tf.transform.rotation.z = transform.rotation.z
    new_tf.transform.rotation.w = transform.rotation.w

    return new_tf

def id_to_short_code(id):
    """Convert a unique id to a 2 letter short code."""
    tokens = id.split('-')
    if len(tokens) > 2:
        return '%c%c' % (tokens[0][0], tokens[1][0])
    return None

def pretty_print_waypoints(waypoint_id, waypoint_name, short_code_to_count, localization_id, logger):
    short_code = id_to_short_code(waypoint_id)
    if short_code is None or short_code_to_count[short_code] != 1:
        short_code = '  '  # If the short code is not valid/unique, don't show it.

    logger.info("%s Waypoint name: %s id: %s short code: %s" %
            ('->' if localization_id == waypoint_id else '  ',
            waypoint_name, waypoint_id, short_code))

def find_unique_waypoint_id(short_code, graph, name_to_id, logger):
    """Convert either a 2 letter short code or an annotation name into the associated unique id."""
    if len(short_code) != 2:
        # Not a short code, check if it is an annotation name (instead of the waypoint id).
        if short_code in name_to_id:
            # Short code is an waypoint's annotation name. Check if it is paired with a unique waypoint id.
            if name_to_id[short_code] is not None:
                # Has an associated waypoint id!
                return name_to_id[short_code]
            else:
                logger.error("The waypoint name %s is used for multiple different unique waypoints. Please use" + \
                        "the waypoint id." % (short_code))
                return None
        # Also not an waypoint annotation name, so we will operate under the assumption that it is a
        # unique waypoint id.
        return short_code

    ret = short_code
    for waypoint in graph.waypoints:
        if short_code == id_to_short_code(waypoint.id):
            if ret != short_code:
                return short_code  # Multiple waypoints with same short code.
            ret = waypoint.id
    return ret

def update_waypoints_and_edges(graph, localization_id, logger):
    """Update and print waypoint ids and edge ids."""
    name_to_id = dict()
    edges = dict()

    short_code_to_count = {}
    waypoint_to_timestamp = []
    for waypoint in graph.waypoints:
        # Determine the timestamp that this waypoint was created at.
        timestamp = -1.0
        try:
            timestamp = waypoint.annotations.creation_time.seconds + waypoint.annotations.creation_time.nanos / 1e9
        except:
            # Must be operating on an older graph nav map, since the creation_time is not
            # available within the waypoint annotations message.
            pass
        waypoint_to_timestamp.append((waypoint.id,
                                        timestamp,
                                        waypoint.annotations.name))

        # Determine how many waypoints have the same short code.
        short_code = id_to_short_code(waypoint.id)
        if short_code not in short_code_to_count:
            short_code_to_count[short_code] = 0
        short_code_to_count[short_code] += 1

        # Add the annotation name/id into the current dictionary.
        waypoint_name = waypoint.annotations.name
        if waypoint_name:
            if waypoint_name in name_to_id:
                # Waypoint name is used for multiple different waypoints, so set the waypoint id
                # in this dictionary to None to avoid confusion between two different waypoints.
                name_to_id[waypoint_name] = None
            else:
                # First time we have seen this waypoint annotation name. Add it into the dictionary
                # with the respective waypoint unique id.
                name_to_id[waypoint_name] = waypoint.id

    # Sort the set of waypoints by their creation timestamp. If the creation timestamp is unavailable,
    # fallback to sorting by annotation name.
    waypoint_to_timestamp = sorted(waypoint_to_timestamp, key= lambda x:(x[1], x[2]))

    # Print out the waypoints name, id, and short code in a ordered sorted by the timestamp from
    # when the waypoint was created.
    logger.info('%d waypoints:' % len(graph.waypoints))
    for waypoint in waypoint_to_timestamp:
        pretty_print_waypoints(waypoint[0], waypoint[2], short_code_to_count, localization_id, logger)

    for edge in graph.edges:
        if edge.id.to_waypoint in edges:
            if edge.id.from_waypoint not in edges[edge.id.to_waypoint]:
                edges[edge.id.to_waypoint].append(edge.id.from_waypoint)
        else:
            edges[edge.id.to_waypoint] = [edge.id.from_waypoint]
        logger.info("(Edge) from waypoint id: ", edge.id.from_waypoint, " and to waypoint id: ",
                edge.id.to_waypoint)

    return name_to_id, edges

class SpotROS():
    """Parent class for using the wrapper.  Defines all callbacks and keeps the wrapper alive"""

    def __init__(self):
        self.spot_wrapper = None

        self.callbacks = {}
        """Dictionary listing what callback to use for what data task"""
        self.callbacks["robot_state"] = self.RobotStateCB
        self.callbacks["metrics"] = self.MetricsCB
        self.callbacks["lease"] = self.LeaseCB
        self.callbacks["front_image"] = self.FrontImageCB
        self.callbacks["side_image"] = self.SideImageCB
        self.callbacks["rear_image"] = self.RearImageCB

    def RobotStateCB(self, results):
        """Callback for when the Spot Wrapper gets new robot state data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        state = self.spot_wrapper.robot_state

        if state:
            ## joint states ##
            joint_state = JointState()
            local_time = self.spot_wrapper.robotToLocalTime(state.kinematic_state.acquisition_timestamp)
            joint_state.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
            for joint in state.kinematic_state.joint_states:
                joint_state.name.append(friendly_joint_names.get(joint.name, "ERROR"))
                joint_state.position.append(joint.position.value)
                joint_state.velocity.append(joint.velocity.value)
                joint_state.effort.append(joint.load.value)
            self.joint_state_pub.publish(joint_state)

            ## TF ##
            tf_msg = TFMessage()
            for frame_name in state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
                if state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name:
                    try:
                        transform = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
                        local_time = self.spot_wrapper.robotToLocalTime(state.kinematic_state.acquisition_timestamp)
                        tf_time = rospy.Time(local_time.seconds, local_time.nanos)
                        if self.inverse_target_frame == frame_name:
                            geo_tform_inversed = SE3Pose.from_obj(transform.parent_tform_child).inverse()
                            new_tf = populateTransformStamped(tf_time, frame_name, transform.parent_frame_name, geo_tform_inversed)
                        else:
                            new_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name, transform.parent_tform_child)
                        tf_msg.transforms.append(new_tf)
                    except Exception as e:
                        self.spot_wrapper.logger.error('Error: {}'.format(e))

            if len(tf_msg.transforms) > 0:
                self.tf_pub.publish(tf_msg)

            # Odom Twist #
            twist_odom_msg = TwistWithCovarianceStamped()
            local_time = self.spot_wrapper.robotToLocalTime(state.kinematic_state.acquisition_timestamp)
            twist_odom_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
            twist_odom_msg.twist.twist.linear.x = state.kinematic_state.velocity_of_body_in_odom.linear.x
            twist_odom_msg.twist.twist.linear.y = state.kinematic_state.velocity_of_body_in_odom.linear.y
            twist_odom_msg.twist.twist.linear.z = state.kinematic_state.velocity_of_body_in_odom.linear.z
            twist_odom_msg.twist.twist.angular.x = state.kinematic_state.velocity_of_body_in_odom.angular.x
            twist_odom_msg.twist.twist.angular.y = state.kinematic_state.velocity_of_body_in_odom.angular.y
            twist_odom_msg.twist.twist.angular.z = state.kinematic_state.velocity_of_body_in_odom.angular.z
            self.odom_twist_pub.publish(twist_odom_msg)

            def GetOdomFromState(state, spot_wrapper, use_vision=True):
                """Maps odometry data from robot state proto to ROS Odometry message

                Args:
                    data: Robot State proto
                    spot_wrapper: A SpotWrapper object
                Returns:
                    Odometry message
                """
                odom_msg = Odometry()
                local_time = spot_wrapper.robotToLocalTime(state.kinematic_state.acquisition_timestamp)
                odom_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
                if use_vision == True:
                    odom_msg.header.frame_id = 'vision'
                    tform_body = get_vision_tform_body(state.kinematic_state.transforms_snapshot)
                else:
                    odom_msg.header.frame_id = 'odom'
                    tform_body = get_odom_tform_body(state.kinematic_state.transforms_snapshot)
                odom_msg.child_frame_id = 'body'
                pose_odom_msg = PoseWithCovariance()
                pose_odom_msg.pose.position.x = tform_body.position.x
                pose_odom_msg.pose.position.y = tform_body.position.y
                pose_odom_msg.pose.position.z = tform_body.position.z
                pose_odom_msg.pose.orientation.x = tform_body.rotation.x
                pose_odom_msg.pose.orientation.y = tform_body.rotation.y
                pose_odom_msg.pose.orientation.z = tform_body.rotation.z
                pose_odom_msg.pose.orientation.w = tform_body.rotation.w

                odom_msg.pose = pose_odom_msg
                twist_odom_msg = TwistWithCovariance()
                twist_odom_msg.twist.linear.x = state.kinematic_state.velocity_of_body_in_odom.linear.x
                twist_odom_msg.twist.linear.y = state.kinematic_state.velocity_of_body_in_odom.linear.y
                twist_odom_msg.twist.linear.z = state.kinematic_state.velocity_of_body_in_odom.linear.z
                twist_odom_msg.twist.angular.x = state.kinematic_state.velocity_of_body_in_odom.angular.x
                twist_odom_msg.twist.angular.y = state.kinematic_state.velocity_of_body_in_odom.angular.y
                twist_odom_msg.twist.angular.z = state.kinematic_state.velocity_of_body_in_odom.angular.z
                odom_msg.twist = twist_odom_msg
                return odom_msg
            
            # Odom #
            if self.mode_parent_odom_tf == 'vision':
                odom_msg = GetOdomFromState(state, self.spot_wrapper, use_vision=True)
            else:
                odom_msg = GetOdomFromState(state, self.spot_wrapper, use_vision=False)
            self.odom_pub.publish(odom_msg)

            # Feet #
            foot_array_msg = FootStateArray()
            for foot in state.foot_state:
                foot_msg = FootState()
                foot_msg.foot_position_rt_body.x = foot.foot_position_rt_body.x
                foot_msg.foot_position_rt_body.y = foot.foot_position_rt_body.y
                foot_msg.foot_position_rt_body.z = foot.foot_position_rt_body.z
                foot_msg.contact = foot.contact
                foot_array_msg.states.append(foot_msg)

            self.feet_pub.publish(foot_array_msg)

            # EStop #
            # TODO: derer to robot state
            estop_array_msg = EStopStateArray()
            for estop in state.estop_states:
                estop_msg = EStopState()
                local_time = self.spot_wrapper.robotToLocalTime(estop.timestamp)
                estop_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
                estop_msg.name = estop.name
                estop_msg.type = estop.type
                estop_msg.state = estop.state
                estop_msg.state_description = estop.state_description
                estop_array_msg.estop_states.append(estop_msg)
            self.estop_pub.publish(estop_array_msg)

            # WIFI #
            wifi_msg = WiFiState()
            for comm_state in state.comms_states:
                if comm_state.HasField('wifi_state'):
                    wifi_msg.current_mode = comm_state.wifi_state.current_mode
                    wifi_msg.essid = comm_state.wifi_state.essid

            self.wifi_pub.publish(wifi_msg)

            # Battery States #
            battery_states_array_msg = BatteryStateArray()
            for battery in state.battery_states:
                battery_msg = BatteryState()
                local_time = self.spot_wrapper.robotToLocalTime(battery.timestamp)
                battery_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
                battery_msg.identifier = battery.identifier
                battery_msg.charge_percentage = battery.charge_percentage.value
                battery_msg.estimated_runtime = rospy.Time(battery.estimated_runtime.seconds, battery.estimated_runtime.nanos)
                battery_msg.current = battery.current.value
                battery_msg.voltage = battery.voltage.value
                for temp in battery.temperatures:
                    battery_msg.temperatures.append(temp)
                battery_msg.status = battery.status
                battery_states_array_msg.battery_states.append(battery_msg)
            self.battery_pub.publish(battery_states_array_msg)

            # # Power State #
            # TODO: derer to robot state
            power_state_msg = PowerState()
            local_time = self.spot_wrapper.robotToLocalTime(state.power_state.timestamp)
            power_state_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
            power_state_msg.motor_power_state = state.power_state.motor_power_state
            power_state_msg.shore_power_state = state.power_state.shore_power_state
            power_state_msg.locomotion_charge_percentage = state.power_state.locomotion_charge_percentage.value
            power_state_msg.locomotion_estimated_runtime = rospy.Time(state.power_state.locomotion_estimated_runtime.seconds, state.power_state.locomotion_estimated_runtime.nanos)
            self.power_pub.publish(power_state_msg)

            def getSystemFaults(system_faults, spot_wrapper):
                """Helper function to strip out system faults into a list

                Args:
                    systen_faults: List of SystemFaults
                    spot_wrapper: A SpotWrapper object
                Returns:
                    List of SystemFault messages
                """
                faults = []

                for fault in system_faults:
                    new_fault = SystemFault()
                    new_fault.name = fault.name
                    local_time = spot_wrapper.robotToLocalTime(fault.onset_timestamp)
                    new_fault.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
                    new_fault.duration = rospy.Time(fault.duration.seconds, fault.duration.nanos)
                    new_fault.code = fault.code
                    new_fault.uid = fault.uid
                    new_fault.error_message = fault.error_message

                    for att in fault.attributes:
                        new_fault.attributes.append(att)

                    new_fault.severity = fault.severity
                    faults.append(new_fault)

                return faults
            
            # System Faults #
            system_fault_state_msg = SystemFaultState()
            system_fault_state_msg.faults = getSystemFaults(state.system_fault_state.faults, self.spot_wrapper)
            system_fault_state_msg.historical_faults = getSystemFaults(state.system_fault_state.historical_faults, self.spot_wrapper)
            self.system_faults_pub.publish(system_fault_state_msg)

            # Behavior Faults #
            behavior_fault_state_msg = BehaviorFaultState()
            faults = []

            for fault in self.behavior_faults:
                new_fault = BehaviorFault()
                new_fault.behavior_fault_id = fault.behavior_fault_id
                local_time = self.spot_wrapper.robotToLocalTime(fault.onset_timestamp)
                new_fault.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
                new_fault.cause = fault.cause
                new_fault.status = fault.status
                faults.append(new_fault)

            self.behavior_faults_pub.publish(behavior_fault_state_msg)

    def MetricsCB(self, results):
        """Callback for when the Spot Wrapper gets new metrics data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        metrics = self.spot_wrapper.metrics
        if metrics:
            metrics_msg = Metrics()
            local_time = self.spot_wrapper.robotToLocalTime(metrics.timestamp)
            metrics_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)

            for metric in metrics.metrics:
                if metric.label == "distance":
                    metrics_msg.distance = metric.float_value
                if metric.label == "gait cycles":
                    metrics_msg.gait_cycles = metric.int_value
                if metric.label == "time moving":
                    metrics_msg.time_moving = rospy.Time(metric.duration.seconds, metric.duration.nanos)
                if metric.label == "electric power":
                    metrics_msg.electric_power = rospy.Time(metric.duration.seconds, metric.duration.nanos)

            self.metrics_pub.publish(metrics_msg)

    def LeaseCB(self, results):
        """Callback for when the Spot Wrapper gets new lease data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        lease_array_msg = LeaseArray()
        lease_list = self.spot_wrapper.lease
        if lease_list:
            for resource in lease_list:
                new_resource = LeaseResource()
                new_resource.resource = resource.resource
                new_resource.lease.resource = resource.lease.resource
                new_resource.lease.epoch = resource.lease.epoch

                for seq in resource.lease.sequence:
                    new_resource.lease.sequence.append(seq)

                new_resource.lease_owner.client_name = resource.lease_owner.client_name
                new_resource.lease_owner.user_name = resource.lease_owner.user_name

                lease_array_msg.resources.append(new_resource)

            self.lease_pub.publish(lease_array_msg)

    def FrontImageCB(self, results):
        """Callback for when the Spot Wrapper gets new front image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        data = self.spot_wrapper.front_images
        if data:
            image_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.frontleft_image_pub.publish(image_msg0)
            self.frontleft_image_info_pub.publish(camera_info_msg0)
            image_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.frontright_image_pub.publish(image_msg1)
            self.frontright_image_info_pub.publish(camera_info_msg1)
            image_msg2, camera_info_msg2 = getImageMsg(data[2], self.spot_wrapper)
            self.frontleft_depth_pub.publish(image_msg2)
            self.frontleft_depth_info_pub.publish(camera_info_msg2)
            image_msg3, camera_info_msg3 = getImageMsg(data[3], self.spot_wrapper)
            self.frontright_depth_pub.publish(image_msg3)
            self.frontright_depth_info_pub.publish(camera_info_msg3)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])
            self.populate_camera_static_transforms(data[2])
            self.populate_camera_static_transforms(data[3])

    def SideImageCB(self, results):
        """Callback for when the Spot Wrapper gets new side image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        data = self.spot_wrapper.side_images
        if data:
            image_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.left_image_pub.publish(image_msg0)
            self.left_image_info_pub.publish(camera_info_msg0)
            image_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.right_image_pub.publish(image_msg1)
            self.right_image_info_pub.publish(camera_info_msg1)
            image_msg2, camera_info_msg2 = getImageMsg(data[2], self.spot_wrapper)
            self.left_depth_pub.publish(image_msg2)
            self.left_depth_info_pub.publish(camera_info_msg2)
            image_msg3, camera_info_msg3 = getImageMsg(data[3], self.spot_wrapper)
            self.right_depth_pub.publish(image_msg3)
            self.right_depth_info_pub.publish(camera_info_msg3)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])
            self.populate_camera_static_transforms(data[2])
            self.populate_camera_static_transforms(data[3])

    def RearImageCB(self, results):
        """Callback for when the Spot Wrapper gets new rear image data.

        Args:
            results: FutureWrapper object of AsyncPeriodicQuery callback
        """
        data = self.spot_wrapper.rear_images
        if data:
            mage_msg0, camera_info_msg0 = getImageMsg(data[0], self.spot_wrapper)
            self.back_image_pub.publish(mage_msg0)
            self.back_image_info_pub.publish(camera_info_msg0)
            mage_msg1, camera_info_msg1 = getImageMsg(data[1], self.spot_wrapper)
            self.back_depth_pub.publish(mage_msg1)
            self.back_depth_info_pub.publish(camera_info_msg1)

            self.populate_camera_static_transforms(data[0])
            self.populate_camera_static_transforms(data[1])

    def handle_stair_mode(self, req):
        """ROS service handler to set a stair mode to the robot."""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.stair_hint = req.data
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetBoolResponse(True, 'Success')
        except Exception as e:
            return SetBoolResponse(False, 'Error:{}'.format(e))

    def handle_locomotion_mode(self, req):
        """ROS service handler to set locomotion mode"""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.locomotion_hint = req.locomotion_mode
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetLocomotionResponse(True, 'Success')
        except Exception as e:
            return SetLocomotionResponse(False, 'Error:{}'.format(e))

    def handle_max_vel(self, req):
        """
        Handle a max_velocity service call. This will modify the mobility params to have a limit on the maximum
        velocity that the robot can move during motion commmands. This affects trajectory commands and velocity
        commands

        Args:
            req: SetVelocityRequest containing requested maximum velocity

        Returns: SetVelocityResponse
        """
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.vel_limit.CopyFrom(SE2VelocityLimit(max_vel=math_helpers.SE2Velocity(req.velocity_limit.linear.x,
                                                                                                 req.velocity_limit.linear.y,
                                                                                                 req.velocity_limit.angular.z).to_proto()))
            self.spot_wrapper.set_mobility_params(mobility_params)
            return SetVelocityResponse(True, 'Success')
        except Exception as e:
            return SetVelocityResponse(False, 'Error:{}'.format(e))

    def handle_trajectory(self, req):
        """ROS actionserver execution handler to handle receiving a request to move to a location"""
        if req.target_pose.header.frame_id != 'body':
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'frame_id of target_pose must be \'body\''))
            return
        if req.duration.data.to_sec() <= 0:
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'duration must be larger than 0'))
            return

        cmd_duration = rospy.Duration(req.duration.data.secs, req.duration.data.nsecs)
        resp = self.spot_wrapper.trajectory_cmd(
                        goal_x=req.target_pose.pose.position.x,
                        goal_y=req.target_pose.pose.position.y,
                        goal_heading=math_helpers.Quat(
                            w=req.target_pose.pose.orientation.w,
                            x=req.target_pose.pose.orientation.x,
                            y=req.target_pose.pose.orientation.y,
                            z=req.target_pose.pose.orientation.z
                            ).to_yaw(),
                        cmd_duration=cmd_duration.to_sec(),
                        precise_position=req.precise_positioning,
                        )

        def timeout_cb(trajectory_server, _):
            trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal, timed out"))
            trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal, timed out"))

        # Abort the actionserver if cmd_duration is exceeded - the driver stops but does not provide feedback to
        # indicate this so we monitor it ourselves
        cmd_timeout = rospy.Timer(cmd_duration, functools.partial(timeout_cb, self.trajectory_server), oneshot=True)

        # The trajectory command is non-blocking but we need to keep this function up in order to interrupt if a
        # preempt is requested and to return success if/when the robot reaches the goal. Also check the is_active to
        # monitor whether the timeout_cb has already aborted the command
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.trajectory_server.is_preempt_requested() and not self.spot_wrapper.at_goal and self.trajectory_server.is_active():
            if self.spot_wrapper.near_goal:
                if self.spot_wrapper._last_trajectory_command_precise:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal, performing final adjustments"))
                else:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal"))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Moving to goal"))
            rate.sleep()

        # If still active after exiting the loop, the command did not time out
        if self.trajectory_server.is_active():
            cmd_timeout.shutdown()
            if self.trajectory_server.is_preempt_requested():
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Preempted"))
                self.trajectory_server.set_preempted()
                self.spot_wrapper.handle_stop()

            if self.spot_wrapper.at_goal:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Reached goal"))
                self.trajectory_server.set_succeeded(TrajectoryResult(resp[0], resp[1]))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal"))
                self.trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal"))

    def cmdVelCallback(self, data):
        """Callback for cmd_vel command"""
        self.spot_wrapper.velocity_cmd(data.linear.x, data.linear.y, data.angular.z)

    def bodyPoseCallback(self, data):
        """Callback for cmd_vel command"""
        q = Quaternion()
        q.x = data.orientation.x
        q.y = data.orientation.y
        q.z = data.orientation.z
        q.w = data.orientation.w
        position = geometry_pb2.Vec3(z=data.position.z)
        pose = geometry_pb2.SE3Pose(position=position, rotation=q)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

        mobility_params = self.spot_wrapper.get_mobility_params()
        mobility_params.body_control.CopyFrom(body_control)
        self.spot_wrapper.set_mobility_params(mobility_params)

    def handle_navigate_to_feedback(self):
        """Thread function to send navigate_to feedback"""
        while not rospy.is_shutdown() and self.run_navigate_to:
            localization_state = self.spot_wrapper._graph_nav_client.get_localization_state()
            if localization_state.localization.waypoint_id:
                self.navigate_as.publish_feedback(NavigateToFeedback(localization_state.localization.waypoint_id))
            rospy.Rate(10).sleep()

    def handle_navigate_to(self, msg):
        """ROS service handler to run mission of the robot.  The robot will replay a mission"""
        # create thread to periodically publish feedback
        feedback_thraed = threading.Thread(target = self.handle_navigate_to_feedback, args = ())
        self.run_navigate_to = True
        feedback_thraed.start()
        # run navigate_to
        resp = self.spot_wrapper.navigate_to(upload_path = msg.upload_path,
                                             navigate_to = msg.navigate_to,
                                             initial_localization_fiducial = msg.initial_localization_fiducial,
                                             initial_localization_waypoint = msg.initial_localization_waypoint)
        self.run_navigate_to = False
        feedback_thraed.join()

        # check status
        if resp[0]:
            self.navigate_as.set_succeeded(NavigateToResult(resp[0], resp[1]))
        else:
            self.navigate_as.set_aborted(NavigateToResult(resp[0], resp[1]))

    def populate_camera_static_transforms(self, image_data):
        """Check data received from one of the image tasks and use the transform snapshot to extract the camera frame
        transforms. This is the transforms from body->frontleft->frontleft_fisheye, for example. These transforms
        never change, but they may be calibrated slightly differently for each robot so we need to generate the
        transforms at runtime.

        Args:
        image_data: Image protobuf data from the wrapper
        """
        # We exclude the odometry frames from static transforms since they are not static. We can ignore the body
        # frame because it is a child of odom or vision depending on the mode_parent_odom_tf, and will be published
        # by the non-static transform publishing that is done by the state callback
        excluded_frames = [self.tf_name_vision_odom, self.tf_name_kinematic_odom, "body"]
        for frame_name in image_data.shot.transforms_snapshot.child_to_parent_edge_map:
            if frame_name in excluded_frames:
                continue
            parent_frame = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
            existing_transforms = [(transform.header.frame_id, transform.child_frame_id) for transform in self.camera_static_transforms]
            if (parent_frame, frame_name) in existing_transforms:
                # We already extracted this transform
                continue

            transform = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
            local_time = self.spot_wrapper.robotToLocalTime(image_data.shot.acquisition_time)
            tf_time = rospy.Time(local_time.seconds, local_time.nanos)
            static_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name,
                                                 transform.parent_tform_child)
            self.camera_static_transforms.append(static_tf)
            self.camera_static_transform_broadcaster.sendTransform(self.camera_static_transforms)

    def setupPubSub(self):
        # Images #
        self.back_image_pub = rospy.Publisher('camera/back/image', Image, queue_size=10)
        self.frontleft_image_pub = rospy.Publisher('camera/frontleft/image', Image, queue_size=10)
        self.frontright_image_pub = rospy.Publisher('camera/frontright/image', Image, queue_size=10)
        self.left_image_pub = rospy.Publisher('camera/left/image', Image, queue_size=10)
        self.right_image_pub = rospy.Publisher('camera/right/image', Image, queue_size=10)
        # Depth #
        self.back_depth_pub = rospy.Publisher('depth/back/image', Image, queue_size=10)
        self.frontleft_depth_pub = rospy.Publisher('depth/frontleft/image', Image, queue_size=10)
        self.frontright_depth_pub = rospy.Publisher('depth/frontright/image', Image, queue_size=10)
        self.left_depth_pub = rospy.Publisher('depth/left/image', Image, queue_size=10)
        self.right_depth_pub = rospy.Publisher('depth/right/image', Image, queue_size=10)

        # Image Camera Info #
        self.back_image_info_pub = rospy.Publisher('camera/back/camera_info', CameraInfo, queue_size=10)
        self.frontleft_image_info_pub = rospy.Publisher('camera/frontleft/camera_info', CameraInfo, queue_size=10)
        self.frontright_image_info_pub = rospy.Publisher('camera/frontright/camera_info', CameraInfo, queue_size=10)
        self.left_image_info_pub = rospy.Publisher('camera/left/camera_info', CameraInfo, queue_size=10)
        self.right_image_info_pub = rospy.Publisher('camera/right/camera_info', CameraInfo, queue_size=10)
        # Depth Camera Info #
        self.back_depth_info_pub = rospy.Publisher('depth/back/camera_info', CameraInfo, queue_size=10)
        self.frontleft_depth_info_pub = rospy.Publisher('depth/frontleft/camera_info', CameraInfo, queue_size=10)
        self.frontright_depth_info_pub = rospy.Publisher('depth/frontright/camera_info', CameraInfo, queue_size=10)
        self.left_depth_info_pub = rospy.Publisher('depth/left/camera_info', CameraInfo, queue_size=10)
        self.right_depth_info_pub = rospy.Publisher('depth/right/camera_info', CameraInfo, queue_size=10)

        # Status Publishers #
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        """Defining a TF publisher manually because of conflicts between Python3 and tf"""
        self.tf_pub = rospy.Publisher('tf', TFMessage, queue_size=10)
        self.metrics_pub = rospy.Publisher('status/metrics', Metrics, queue_size=10)
        self.lease_pub = rospy.Publisher('status/leases', LeaseArray, queue_size=10)
        self.odom_twist_pub = rospy.Publisher('odometry/twist', TwistWithCovarianceStamped, queue_size=10)
        self.odom_pub = rospy.Publisher('odometry', Odometry, queue_size=10)
        self.feet_pub = rospy.Publisher('status/feet', FootStateArray, queue_size=10)
        self.wifi_pub = rospy.Publisher('status/wifi', WiFiState, queue_size=10)
        self.battery_pub = rospy.Publisher('status/battery_states', BatteryStateArray, queue_size=10)
        self.behavior_faults_pub = rospy.Publisher('status/behavior_faults', BehaviorFaultState, queue_size=10)
        self.system_faults_pub = rospy.Publisher('status/system_faults', SystemFaultState, queue_size=10)

        self.feedback_pub = rospy.Publisher('status/feedback', Feedback, queue_size=10)

        self.mobility_params_pub = rospy.Publisher('status/mobility_params', MobilityParams, queue_size=10)

        rospy.Subscriber('cmd_vel', Twist, self.cmdVelCallback, queue_size = 1)
        rospy.Subscriber('body_pose', Pose, self.bodyPoseCallback, queue_size = 1)
        rospy.Service("stair_mode", SetBool, self.handle_stair_mode)
        rospy.Service("locomotion_mode", SetLocomotion, self.handle_locomotion_mode)
        rospy.Service("max_velocity", SetVelocity, self.handle_max_vel)

        self.navigate_as = actionlib.SimpleActionServer('navigate_to', NavigateToAction,
                                                        execute_cb = self.handle_navigate_to,
                                                        auto_start = False)
        self.navigate_as.start()

        self.trajectory_server = actionlib.SimpleActionServer("trajectory", TrajectoryAction,
                                                                execute_cb=self.handle_trajectory,
                                                                auto_start=False)
        self.trajectory_server.start()

    def main(self):
        """Main function for the SpotROS class.  Gets config from ROS and initializes the wrapper.  Holds lease from wrapper and updates all async tasks at the ROS rate"""
        rospy.init_node('spot_ros', anonymous=True)
        rate = rospy.Rate(50)

        self.rates = rospy.get_param('~rates', {})
        self.username = rospy.get_param('~username', 'default_value')
        self.password = rospy.get_param('~password', 'default_value')
        self.hostname = rospy.get_param('~hostname', 'default_value')
        self.motion_deadzone = rospy.get_param('~deadzone', 0.05)

        self.camera_static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # Static transform broadcaster is super simple and just a latched publisher. Every time we add a new static
        # transform we must republish all static transforms from this source, otherwise the tree will be incomplete.
        # We keep a list of all the static transforms we already have so they can be republished, and so we can check
        # which ones we already have
        self.camera_static_transforms = []

        # Spot has 2 types of odometries: 'odom' and 'vision'
        # The former one is kinematic odometry and the second one is a combined odometry of vision and kinematics
        # These params enables to change which odometry frame is a parent of body frame and to change tf names of each odometry frames.
        self.mode_parent_odom_tf = rospy.get_param('~mode_parent_odom_tf', 'odom') # 'vision' or 'odom'
        self.tf_name_kinematic_odom = rospy.get_param('~tf_name_kinematic_odom', 'odom')
        self.tf_name_raw_kinematic = 'odom'
        self.tf_name_vision_odom = rospy.get_param('~tf_name_vision_odom', 'vision')
        self.tf_name_raw_vision = 'vision'
        if self.mode_parent_odom_tf != self.tf_name_raw_kinematic and self.mode_parent_odom_tf != self.tf_name_raw_vision:
            rospy.logerr('rosparam \'~mode_parent_odom_tf\' should be \'odom\' or \'vision\'.')
            return

        self.logger = logging.getLogger('rosout')

        rospy.logwarn("Starting ROS driver for Spot")
        rospy.logwarn("Getting spot wrapper...")
        self.spot_wrapper = SpotWrapper(self.username, self.password, self.hostname, self.logger, self.rates, self.callbacks)
        rospy.logwarn("Acquired spot wrapper.")
        if self.spot_wrapper.is_valid:
            self.setupPubSub()
            rospy.on_shutdown(self.shutdown)

            self.auto_claim = rospy.get_param('~auto_claim', False)
            self.auto_power_on = rospy.get_param('~auto_power_on', False)
            self.auto_stand = rospy.get_param('~auto_stand', False)

            try:
                if self.auto_claim:
                    success, msg = self.spot_wrapper.claim()
                    if not success:
                        rospy.logerr('Unable to claim spot: ' + msg)
                        rospy.signal_shutdown("Unable to claim spot: graceful shutdown")
                    if self.auto_power_on:
                        success, msg = self.spot_wrapper.power_on()
                        if not success:
                            rospy.logerr('Unable to power on: ' + msg)
                            rospy.signal_shutdown("Unable to power on: graceful shutdown")
                        if self.auto_stand:
                            self.spot_wrapper.stand()
            except Exception as e:
                rospy.logerr('Unknown exception: ')
                rospy.logerr(e)
                os._exit(1)

            while not rospy.is_shutdown():
                """Loop through all periodic tasks and update their data if needed."""
                try:
                    self._async_tasks.update()
                except Exception as e:
                    print(f"Update tasks failed with error: {str(e)}")

                feedback_msg = Feedback()
                feedback_msg.standing = self.spot_wrapper.is_standing
                feedback_msg.sitting = self.spot_wrapper.is_sitting
                feedback_msg.moving = self.spot_wrapper.is_moving
                id = self.spot_wrapper.id
                try:
                    feedback_msg.serial_number = id.serial_number
                    feedback_msg.species = id.species
                    feedback_msg.version = id.version
                    feedback_msg.nickname = id.nickname
                    feedback_msg.computer_serial_number = id.computer_serial_number
                except:
                    pass
                self.feedback_pub.publish(feedback_msg)
                mobility_params_msg = MobilityParams()
                try:
                    mobility_params = self.spot_wrapper.get_mobility_params()
                    mobility_params_msg.body_control.position.x = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.x
                    mobility_params_msg.body_control.position.y = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.y
                    mobility_params_msg.body_control.position.z = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.position.z
                    mobility_params_msg.body_control.orientation.x = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.x
                    mobility_params_msg.body_control.orientation.y = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.y
                    mobility_params_msg.body_control.orientation.z = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.z
                    mobility_params_msg.body_control.orientation.w = \
                            mobility_params.body_control.base_offset_rt_footprint.points[0].pose.rotation.w
                    mobility_params_msg.locomotion_hint = mobility_params.locomotion_hint
                    mobility_params_msg.stair_hint = mobility_params.stair_hint
                except Exception as e:
                    rospy.logerr('Error:{}'.format(e))
                    pass
                self.mobility_params_pub.publish(mobility_params_msg)
                rate.sleep()
        else:
            rospy.logerror("Invalid spot wrapper!")

class MinimalSpotROS:
    def __init__(self, username, password, hostname, logging_level=0):
        # TODO: no logging, no error handling for now

        # ---- BD comms ----
        self._sdk = create_standard_sdk('ros_spot')
        self._robot = self._sdk.create_robot(self._hostname)
        self._robot.authenticate(self._username, self._password)
        self._robot.start_time_sync()
        self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # ---- ROS ----
        # E-Stop
        self.estop_pub = rospy.Publisher('status/estop', EStopStateArray, queue_size=10)  # publish estop state
        rospy.Service("estop/hard", Trigger, self.handle_estop_hard)  # service to trigger hard/immediate estop
        rospy.Service("estop/gentle", Trigger, self.handle_estop_soft)  # service to trigger soft/safe estop
        rospy.Service("estop/release", Trigger, self.handle_estop_disengage)  # service to release estop: motors allowed

        # Power
        self.power_pub = rospy.Publisher('status/power_state', PowerState, queue_size=10)
        rospy.Service("power_on", Trigger, self.handle_power_on) # service to power on 
        rospy.Service("power_off", Trigger, self.handle_safe_power_off) # service to power off

        # Sit / Stand
        rospy.Service("sit", Trigger, self.handle_sit) # service to trigger sitting
        rospy.Service("stand", Trigger, self.handle_stand) # servive to trigger standing

        rospy.Service("claim", Trigger, self.handle_claim) # service to trigger claim
        rospy.Service("release", Trigger, self.handle_release) # service to trigger release
        rospy.Service("self_right", Trigger, self.handle_self_right) # service to trigger self right

        # Clear_Behavior_Fault
        rospy.Service("clear_behavior_fault", ClearBehaviorFault, self.handle_clear_behavior_fault)

        # List Graph
        rospy.Service("list_graph", ListGraph, self.handle_list_graph)

        self._estop_endpoint, self._estop_keepalive = None, None
        
    def handle_estop_hard(self, _):
        self._estop_keepalive.handle_stop()
        return TriggerResponse(True, "Success")

    def handle_estop_soft(self, _):
        self._estop_keepalive.settle_then_cut()
        return TriggerResponse(True, "Success")

    def handle_estop_disengage(self, _):
        self._estop_keepalive.allow()
        return TriggerResponse(True, "Success")      

    def handle_claim(self, req):
            """ROS service handler for the claim service"""
            try:
                self._robot_id = self._robot.get_id()
                self._lease = self._lease_client.take()
                self._lease_keepalive = LeaseKeepAlive(self._lease_client)

                # TODO: defer to lease claim
                self._estop_endpoint = EstopEndpoint(self._estop_client, 'ros', 9.0)
                self._estop_endpoint.force_simple_setup()  # Set this endpoint as the robot's sole estop.
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
                
                return TriggerResponse(success=True, message="Success")
            except (ResponseError, RpcError) as err:
                self._logger.error("Failed to initialize robot communication: %s", err)
                return TriggerResponse(success=False, message=str(err))

    def handle_release(self, req):
        """ROS service handler for the release service"""
        resp = self.spot_wrapper.release()
        return TriggerResponse(resp[0], resp[1])
    
    def handle_release(self, req):
        """ROS service handler for the release service"""
        try:
            # Combined release and releaseLease logic here
            if self._lease:
                self._lease_client.return_lease(self._lease)
                self._lease = None

            # TODO: defer to lease release
            if self._estop_keepalive:
                self._estop_keepalive.stop()
                self._estop_keepalive = None
                self._estop_endpoint = None

            return TriggerResponse(success=True, message="Success")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))           
        
    def handle_self_right(self, req):
        """ROS service handler for the self-right service"""
        response = self._robot_command(RobotCommandBuilder.selfright_command())
        return TriggerResponse(success=response[0], message=response[1])

    def handle_clear_behavior_fault(self, req):
        """ROS service handler for clearing behavior faults"""
        try:
            rid = self._robot_command_client.clear_behavior_fault(behavior_fault_id=req.id, lease=None)
            return ClearBehaviorFaultResponse(success=True, message="Success", fault_id=rid)
        except Exception as e:
            return ClearBehaviorFaultResponse(success=False, message=str(e), fault_id=None)        

    def handle_list_graph(self, req):
        """ROS service handler for listing graph_nav waypoint_ids"""
        ids, eds = self._list_graph_waypoint_and_edge_ids(req.upload_path)
        waypoint_ids = [v for k, v in sorted(ids.items(), key=lambda id: int(id[0].replace('waypoint_', '')))]
        return ListGraphResponse(waypoint_ids=waypoint_ids)       

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            self._logger.error("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = update_waypoints_and_edges(
            graph, localization_id, self._logger)
        return self._current_annotation_name_to_wp_id, self._current_edges    
        
    def handle_power_on(self, _):
        """ROS service handler for the power-on service"""
        try:
            self.power_on(self._power_client)
            return TriggerResponse(True, "Success")
        except Exception as e:
            return False, str(e)
        
        return TriggerResponse(resp[0], resp[1])

    def handle_safe_power_off(self, _):
        """ROS service handler for the safe-power-off service"""
        self._robot_command(RobotCommandBuilder.safe_power_off_command())
        return TriggerResponse(True, "Success")

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(timeout=10) # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_proto.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on
    
    def shutdown(self):

        rospy.loginfo("Shutting down ROS driver for Spot")

        # If the user requests a shutdown using "rosnode kill", the
        # safe_power_off command will not sit the robot down. Perform a
        # redundant sit to handle this edge case.
        self.spot_wrapper.sit()

        rospy.logwarn("Requesting safe power off...")
        success, msg = self.spot_wrapper.safe_power_off()
        if not success:
            rospy.logerr(f"Unable to perform safe power off: {msg}")
        else:
            # Wait for the robot to fully sit down, power off its motors, and
            # register that its motors are powered off.
            time.sleep(6.0)
            rospy.logwarn("Safely powered off the robot.")

        rospy.logwarn("Releasing Spot Lease and E-Stop authority...")
        success, msg = self.spot_wrapper.release()
        if not success:
            rospy.logerr(f"Unable to release Spot Lease and E-Stop authority: {msg}")

        rospy.logwarn("Released Spot Lease and E-Stop authority.")

    def handle_sit(self, req):
        """Stop the robot's motion and sit down if able."""
        response = self._robot_command(RobotCommandBuilder.synchro_sit_command())
        self._last_sit_command = response[2]
        return response[0], response[1]

    def handle_stand(self, req):
        """If the e-stop is enabled, and the motor power is enabled, stand the robot up."""
        response = self._robot_command(RobotCommandBuilder.synchro_stand_command(params=self._mobility_params))
        if monitor_command:
            self._last_stand_command = response[2]
        return response[0], response[1]

    def handle_stop(self, req):
        """ROS service handler for the stop service"""
        response = self._robot_command(RobotCommandBuilder.stop_command())
        return response[0], response[1]
    
    def handle_stair_mode(self, req):
        """ROS service handler to set a stair mode to the robot."""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.stair_hint = req.data
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetBoolResponse(True, 'Success')
        except Exception as e:
            return SetBoolResponse(False, 'Error:{}'.format(e))
    
    def get_mobility_params(self):
        """Get mobility params
        """
        return self._mobility_params   

    def set_mobility_params(self, mobility_params):
        """Set Params for mobility and movement

        Args:
            mobility_params: spot.MobilityParams, params for spot mobility commands.
        """
        self._mobility_params = mobility_params

    def handle_locomotion_mode(self, req):
        """ROS service handler to set locomotion mode"""
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.locomotion_hint = req.locomotion_mode
            self.spot_wrapper.set_mobility_params( mobility_params )
            return SetLocomotionResponse(True, 'Success')
        except Exception as e:
            return SetLocomotionResponse(False, 'Error:{}'.format(e))

    def handle_max_vel(self, req):
        """
        Handle a max_velocity service call. This will modify the mobility params to have a limit on the maximum
        velocity that the robot can move during motion commmands. This affects trajectory commands and velocity
        commands

        Args:
            req: SetVelocityRequest containing requested maximum velocity

        Returns: SetVelocityResponse
        """
        try:
            mobility_params = self.spot_wrapper.get_mobility_params()
            mobility_params.vel_limit.CopyFrom(SE2VelocityLimit(max_vel=math_helpers.SE2Velocity(req.velocity_limit.linear.x,
                                                                                                 req.velocity_limit.linear.y,
                                                                                                 req.velocity_limit.angular.z).to_proto()))
            self.spot_wrapper.set_mobility_params(mobility_params)
            return SetVelocityResponse(True, 'Success')
        except Exception as e:
            return SetVelocityResponse(False, 'Error:{}'.format(e))

    def handle_trajectory(self, req):
        """ROS actionserver execution handler to handle receiving a request to move to a location"""
        if req.target_pose.header.frame_id != 'body':
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'frame_id of target_pose must be \'body\''))
            return
        if req.duration.data.to_sec() <= 0:
            self.trajectory_server.set_aborted(TrajectoryResult(False, 'duration must be larger than 0'))
            return

        cmd_duration = rospy.Duration(req.duration.data.secs, req.duration.data.nsecs)
        resp = self.spot_wrapper.trajectory_cmd(
                        goal_x=req.target_pose.pose.position.x,
                        goal_y=req.target_pose.pose.position.y,
                        goal_heading=math_helpers.Quat(
                            w=req.target_pose.pose.orientation.w,
                            x=req.target_pose.pose.orientation.x,
                            y=req.target_pose.pose.orientation.y,
                            z=req.target_pose.pose.orientation.z
                            ).to_yaw(),
                        cmd_duration=cmd_duration.to_sec(),
                        precise_position=req.precise_positioning,
                        )

        def timeout_cb(trajectory_server, _):
            trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal, timed out"))
            trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal, timed out"))

        # Abort the actionserver if cmd_duration is exceeded - the driver stops but does not provide feedback to
        # indicate this so we monitor it ourselves
        cmd_timeout = rospy.Timer(cmd_duration, functools.partial(timeout_cb, self.trajectory_server), oneshot=True)

        # The trajectory command is non-blocking but we need to keep this function up in order to interrupt if a
        # preempt is requested and to return success if/when the robot reaches the goal. Also check the is_active to
        # monitor whether the timeout_cb has already aborted the command
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.trajectory_server.is_preempt_requested() and not self.spot_wrapper.at_goal and self.trajectory_server.is_active():
            if self.spot_wrapper.near_goal:
                if self.spot_wrapper._last_trajectory_command_precise:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal, performing final adjustments"))
                else:
                    self.trajectory_server.publish_feedback(TrajectoryFeedback("Near goal"))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Moving to goal"))
            rate.sleep()

        # If still active after exiting the loop, the command did not time out
        if self.trajectory_server.is_active():
            cmd_timeout.shutdown()
            if self.trajectory_server.is_preempt_requested():
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Preempted"))
                self.trajectory_server.set_preempted()
                self.spot_wrapper.handle_stop()

            if self.spot_wrapper.at_goal:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Reached goal"))
                self.trajectory_server.set_succeeded(TrajectoryResult(resp[0], resp[1]))
            else:
                self.trajectory_server.publish_feedback(TrajectoryFeedback("Failed to reach goal"))
                self.trajectory_server.set_aborted(TrajectoryResult(False, "Failed to reach goal"))
   

