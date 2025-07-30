def remove_duplicate_boxes(boxes, compare_single=None, iou_threshold=0.7):
    """
    Removes duplicate or highly overlapping boxes, keeping the larger one.
    :param boxes: List of (x1, y1, x2, y2) boxes.
    :param compare_single: Optional single box to compare against the list.
    :param iou_threshold: IOU threshold to consider as duplicate.
    :return: 
        - If compare_single is None: deduplicated list of boxes.
        - If compare_single is provided: tuple (is_duplicate, updated_box_or_none)
    """
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def compute_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # Single comparison mode
    if compare_single is not None:
        single_area = compute_area(compare_single)
        for existing_box in boxes:
            iou = compute_iou(compare_single, existing_box)
            if iou > iou_threshold:
                existing_area = compute_area(existing_box)
                if single_area > existing_area:
                    return True, compare_single  # Keep new (larger) box
                else:
                    return True, None  # Existing box is better, discard new
        return False, compare_single  # No overlap found, keep it

    # Bulk deduplication mode
    unique_boxes = []
    for box in boxes:
        box_area = compute_area(box)
        replaced_existing = False
        
        # Check against existing unique boxes
        for i, ubox in enumerate(unique_boxes):
            if compute_iou(box, ubox) > iou_threshold:
                ubox_area = compute_area(ubox)
                # If current box is larger, replace the existing one
                if box_area > ubox_area:
                    unique_boxes[i] = box
                    replaced_existing = True
                # If existing box is larger or equal, ignore current box
                break
        
        # If no overlap found, add the box
        if not replaced_existing and not any(compute_iou(box, ubox) > iou_threshold for ubox in unique_boxes):
            unique_boxes.append(box)

    print(f"✅ Found {abs(len(unique_boxes) - len(boxes))} duplicates")
    return unique_boxes