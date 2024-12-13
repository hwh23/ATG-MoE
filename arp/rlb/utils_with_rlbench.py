RLBENCH_18TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg", "light_bulb_in",
    "meat_off_grill", "open_drawer", "place_shape_in_shape_sorter",
    "place_wine_at_rack_location", "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]
# 得到一个字典，如TASK_TO_ID["turn_tap"] 会返回 17
TASK_TO_ID = {task: i for i, task in enumerate(RLBENCH_18TASKS)}
