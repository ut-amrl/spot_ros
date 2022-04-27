# Contact
uint8 CONTACT_UNKNOWN = 0
uint8 CONTACT_MADE = 1
uint8 CONTACT_LOST = 2

geometry_msgs/Point foot_position_rt_body
uint8 contact

# Foot specific terrain data as defined in SDK v3.0.3.
# Data may not be valid if the contact state is not CONTACT_MADE.
# Note that the Boston Dynamics Protobuf definition uses a TerrainState
# submessage instead of storing these fields directly.

# Estimated ground coefficient of friction for this foot.
float64 ground_mu_est
# Reference frame for vector data.
string frame_name
# Foot slip distance rt named frame
geometry_msgs/Point foot_slip_distance_rt_frame
# Foot slip velocity rt named frame
geometry_msgs/Point foot_slip_velocity_rt_frame
# Ground contact normal rt named frame
geometry_msgs/Point ground_contact_normal_rt_frame
# Mean penetration (meters) of the foot below the ground visual
# surface. For penetrable terrains (gravel/sand/grass etc.) these values are
# positive. Negative values would indicate potential odometry issues.
float64 visual_surface_ground_penetration_mean
# Standard deviation of the visual surface ground penetration.
float64 visual_surface_ground_penetration_std
