import grafics

grafics.create_window(1600, 900, [0, 0, 0], 0.0, 0.35, 0.0, 70.0)

grafics.prism(
    color=[0, 255, 0, 255],
    line_color=[0, 0, 0, 255],
    line_width=2, 
    primary=[0.0, 0.0, 1.0],
    world_poses=[
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.0, 2.0]
    ]
)

grafics.start()