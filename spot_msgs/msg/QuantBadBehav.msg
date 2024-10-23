string log
std_msgs/Header header

# level 1: at the interface of NAV and spot-ros
geometry_msgs/Twist cmd_vel
geometry_msgs/Pose body_pose

# level 2: at the interface of spot-ros and bd-sdk (black box)
string command_proto   # has both vel and body_pose
float64 end_time
bool res_bool
string res_str

# level 3: (feeback from env) contact & normal from ground 
