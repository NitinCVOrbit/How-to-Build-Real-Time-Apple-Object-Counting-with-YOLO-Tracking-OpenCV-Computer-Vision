def is_center_crossing_line(prev_center, curr_center, line_p1, line_p2):
    """
    Checks if an object center crossed the line
    by comparing its position in two consecutive frames
    """

    # ---------------------------------------------------------
    # STEP 1: Extract line endpoints
    # Example:
    # line_p1 = (523, 370)
    # line_p2 = (650, 556)
    # This defines a diagonal counting line
    # ---------------------------------------------------------
    x1, y1 = line_p1   # x1=523, y1=370
    x2, y2 = line_p2   # x2=650, y2=556

    # ---------------------------------------------------------
    # STEP 2: Extract previous frame center
    # Example:
    # prev_center = (560, 350)
    # Object is ABOVE the line
    # ---------------------------------------------------------
    px, py = prev_center

    # ---------------------------------------------------------
    # STEP 3: Extract current frame center
    # Example:
    # curr_center = (590, 420)
    # Object moved BELOW the line
    # ---------------------------------------------------------
    cx, cy = curr_center

    # ---------------------------------------------------------
    # STEP 4: Compute which SIDE of the line
    # the previous center lies on
    #
    # Formula:
    # (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    #
    # Plugging values:
    # (650-523)*(py-370) - (556-370)*(px-523)
    # = 127*(py-370) - 186*(px-523)
    #
    # Example:
    # py=350, px=560
    # = 127*(-20) - 186*(37)
    # = -2540 - 6882
    # = -9422  → NEGATIVE SIDE
    # ---------------------------------------------------------
    prev_side = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # ---------------------------------------------------------
    # STEP 5: Compute which SIDE of the line
    # the current center lies on
    #
    # Example:
    # cy=420, cx=590
    # = 127*(420-370) - 186*(590-523)
    # = 127*50 - 186*67
    # = 6350 - 12462
    # = -6112 → STILL NEGATIVE (not crossed yet)
    #
    # If this value becomes POSITIVE → crossing happened
    # ---------------------------------------------------------
    curr_side = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

    # ---------------------------------------------------------
    # STEP 6: Check SIGN CHANGE
    #
    # Case 1:
    # prev_side = -9422
    # curr_side = +3000
    # (-) * (+) < 0 → CROSSED
    #
    # Case 2:
    # prev_side = -9422
    # curr_side = -6112
    # (-) * (-) > 0 → NOT CROSSED
    # ---------------------------------------------------------
    if prev_side * curr_side < 0:
        return True   # Object crossed the line

    # ---------------------------------------------------------
    # STEP 7: Otherwise → no crossing
    # ---------------------------------------------------------
    return False







def get_smooth_center(track_id, new_center):

    # Add the current detected center point to the history of this object
    # Over multiple frames, this history may look like:
    # [(100,200), (102,198), (101,201), (103,199)]
    # These small variations are caused by detection noise (jitter)
    center_history[track_id].append(new_center)

    # Extract all X coordinates from the history
    xs = [c[0] for c in center_history[track_id]]

    # Extract all Y coordinates from the history
    ys = [c[1] for c in center_history[track_id]]

    # Compute average X coordinate
    # Example:
    # xs = [100, 102, 101, 103]
    # avg_x = (100 + 102 + 101 + 103) / 4 = 101.5 → 101
    avg_x = int(sum(xs) / len(xs))

    # Compute average Y coordinate
    # Example:
    # ys = [200, 198, 201, 199]
    # avg_y = (200 + 198 + 201 + 199) / 4 = 199.5 → 199
    avg_y = int(sum(ys) / len(ys))

    # Return the averaged (smoothed) center point instead of the raw center
    # WHY this is important:
    # 1) Reduces sudden jumps caused by detector noise
    # 2) Makes object movement smoother across frames
    # 3) Prevents false line-crossing detections
    # 4) Improves counting accuracy and stability
    return (avg_x, avg_y)
