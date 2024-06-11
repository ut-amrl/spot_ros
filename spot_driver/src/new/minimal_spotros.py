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

# ROS
from spot_msgs.msg import EStopState, EStopStateArray
from std_srvs.srv import Trigger, TriggerResponse

# BD Spot SDK
from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive


class MinimalSpotROS:
    def __init__(self, username, password, hostname, logging_level=0):
        # TODO: no logging, no error handling for now

        # ---- BD comms ----
        self._sdk = create_standard_sdk('ros_spot')
        self._robot = self._sdk.create_robot(self._hostname)
        self._robot.authenticate(self._username, self._password)
        self._robot.start_time_sync()
        self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)

        # ---- ROS ----
        # E-Stop
        self.estop_pub = rospy.Publisher('status/estop', EStopStateArray, queue_size=10)  # publish estop state
        rospy.Service("estop/hard", Trigger, self.handle_estop_hard)  # service to trigger hard/immediate estop
        rospy.Service("estop/gentle", Trigger, self.handle_estop_soft)  # service to trigger soft/safe estop
        rospy.Service("estop/release", Trigger, self.handle_estop_disengage)  # service to release estop: motors allowed

        self._estop_endpoint, self._estop_keepalive = None, None

    def handle_estop_hard(self, _):
        self._estop_keepalive.stop()
        return TriggerResponse(True, "Success")

    def handle_estop_soft(self, _):
        self._estop_keepalive.settle_then_cut()
        return TriggerResponse(True, "Success")

    def handle_estop_disengage(self, _):
        self._estop_keepalive.allow()
        return True, "Success"
