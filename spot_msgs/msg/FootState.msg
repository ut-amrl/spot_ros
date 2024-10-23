# Contact
uint8 CONTACT_UNKNOWN = 0
uint8 CONTACT_MADE = 1
uint8 CONTACT_LOST = 2

# based on bd sdk v4.0.2
# Information about the foot positions and contact state, on a per-foot basis.
geometry_msgs/Vector3 foot_position_rt_body  # The foot position described relative to the body
uint8 contact                                # Is the foot in contact with the ground?

# Foot specific terrain data. Data may not be valid if the contact state is not CONTACT_MADE.
float64 ground_mu_est                        # Estimated ground coefficient of friction for this foot.
string frame_name                            # Reference frame name for vector data.
geometry_msgs/Vector3 foot_slip_distance_rt_frame  # Foot slip distance in the named frame.
geometry_msgs/Vector3 foot_slip_velocity_rt_frame  # Foot slip velocity in the named frame.
geometry_msgs/Vector3 ground_contact_normal_rt_frame  # Ground contact normal in the named frame.
float64 visual_surface_ground_penetration_mean  # Mean penetration (meters) of the foot below the ground visual surface. For penetrable terrains (gravel/sand/grass etc.) these values are positive. Negative values would indicate odometry issues
float64 visual_surface_ground_penetration_std   # Standard deviation of the ground surface penetration.