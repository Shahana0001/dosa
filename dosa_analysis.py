import cv2
import numpy as np
import math


# ============================================================
# DOSA DETECTION
# ============================================================

def detect_dosa(image):
    """
    Detect the main dosa-shaped object in an image.

    Returns:
        annotated_image
        mask
        contour
        detection_info
    """

    original = image.copy()

    # Resize large images for faster processing
    max_width = 1000

    h, w = image.shape[:2]

    if w > max_width:
        scale = max_width / w
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale))
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold for bright dosa region
    _, threshold = cv2.threshold(
        blurred,
        80,
        255,
        cv2.THRESH_BINARY
    )

    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return original, None, None, None

    image_area = image.shape[0] * image.shape[1]

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.03:
            continue

        if area > image_area * 0.90:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * math.pi * area
        ) / (perimeter * perimeter)

        x, y, bw, bh = cv2.boundingRect(contour)

        aspect_ratio = min(bw, bh) / max(bw, bh)

        # Dosa should generally be somewhat circular
        score = (
            area / image_area
            + circularity * 0.5
            + aspect_ratio * 0.3
        )

        candidates.append(
            (score, contour)
        )

    if not candidates:
        return original, None, None, None

    # Best candidate
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    contour = candidates[0][1]

    # Create mask
    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        -1
    )

    # Annotated image
    annotated = image.copy()

    cv2.drawContours(
        annotated,
        [contour],
        -1,
        (0, 255, 0),
        4
    )

    x, y, bw, bh = cv2.boundingRect(contour)

    detection_info = {
        "width": bw,
        "height": bh,
        "area": cv2.contourArea(contour)
    }

    return annotated, mask, contour, detection_info


# ============================================================
# GEOMETRY
# ============================================================

def calculate_geometry(contour, mask):
    """
    Calculate dosa geometry.
    """

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    # Minimum enclosing circle
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(
        contour
    )

    diameter = radius * 2

    # Moments for centroid
    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        centroid_x = (
            moments["m10"] /
            moments["m00"]
        )

        centroid_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        centroid_x = circle_x
        centroid_y = circle_y

    geometry = {
        "area": area,
        "perimeter": perimeter,
        "diameter": diameter,
        "radius": radius,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "circle_center_x": circle_x,
        "circle_center_y": circle_y
    }

    return geometry


# ============================================================
# ROUNDNESS
# ============================================================

def calculate_roundness(area, perimeter):
    """
    Circularity:
        4*pi*Area / Perimeter^2

    Converts circularity into a percentage.
    """

    if perimeter <= 0:
        return 0.0

    circularity = (
        4 * math.pi * area
    ) / (perimeter * perimeter)

    # Limit to valid range
    circularity = max(
        0.0,
        min(1.0, circularity)
    )

    score = circularity * 100

    return round(score, 1)


# ============================================================
# SYMMETRY
# ============================================================

def calculate_symmetry(mask):
    """
    Calculate left-right and top-bottom symmetry
    using the detected dosa mask.
    """

    if mask is None:
        return 0.0, 0.0, 0.0

    # Normalize mask
    mask_binary = (
        mask > 0
    ).astype(np.uint8) * 255

    # Left-right comparison
    flipped_lr = cv2.flip(
        mask_binary,
        1
    )

    intersection_lr = np.logical_and(
        mask_binary > 0,
        flipped_lr > 0
    ).sum()

    union_lr = np.logical_or(
        mask_binary > 0,
        flipped_lr > 0
    ).sum()

    if union_lr > 0:
        lr_score = (
            intersection_lr /
            union_lr
        ) * 100
    else:
        lr_score = 0

    # Top-bottom comparison
    flipped_tb = cv2.flip(
        mask_binary,
        0
    )

    intersection_tb = np.logical_and(
        mask_binary > 0,
        flipped_tb > 0
    ).sum()

    union_tb = np.logical_or(
        mask_binary > 0,
        flipped_tb > 0
    ).sum()

    if union_tb > 0:
        tb_score = (
            intersection_tb /
            union_tb
        ) * 100
    else:
        tb_score = 0

    overall = (
        lr_score + tb_score
    ) / 2

    return (
        round(lr_score, 1),
        round(tb_score, 1),
        round(overall, 1)
    )


# ============================================================
# CENTER ACCURACY
# ============================================================

def calculate_center_accuracy(
    centroid_x,
    centroid_y,
    circle_x,
    circle_y,
    radius
):
    """
    Compare centroid with enclosing circle center.
    """

    distance = math.sqrt(
        (centroid_x - circle_x) ** 2
        +
        (centroid_y - circle_y) ** 2
    )

    if radius <= 0:
        accuracy = 0
    else:

        deviation_ratio = (
            distance / radius
        )

        accuracy = (
            1 - deviation_ratio
        ) * 100

        accuracy = max(
            0,
            min(100, accuracy)
        )

    return (
        round(distance, 2),
        round(accuracy, 1)
    )


# ============================================================
# EDGE QUALITY
# ============================================================

def calculate_edge_quality(
    contour,
    centroid_x,
    centroid_y,
    radius
):
    """
    Measure how consistently the dosa boundary
    follows a circular shape.

    Uses radial distances from the centroid.
    """

    if contour is None or radius <= 0:
        return 0.0, 0.0

    points = contour.reshape(
        -1,
        2
    )

    distances = []

    for x, y in points:

        distance = math.sqrt(
            (x - centroid_x) ** 2
            +
            (y - centroid_y) ** 2
        )

        distances.append(distance)

    distances = np.array(
        distances,
        dtype=np.float32
    )

    if len(distances) == 0:
        return 0.0, 0.0

    mean_distance = np.mean(
        distances
    )

    std_distance = np.std(
        distances
    )

    if mean_distance == 0:
        return 0.0, 0.0

    variation = (
        std_distance /
        mean_distance
    )

    # Lower variation = better edge
    quality = (
        1 -
        min(variation, 1)
    ) * 100

    irregularity = min(
        variation * 100,
        100
    )

    return (
        round(quality, 1),
        round(irregularity, 1)
    )


# ============================================================
# BROWNING ANALYSIS
# ============================================================

def calculate_browning(image, mask):
    """
    Analyze brightness/color variation inside dosa.

    This is a visual image-based estimate.
    It is NOT a temperature measurement.
    """

    if image is None or mask is None:
        return 0.0, None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dosa_pixels = gray[
        mask > 0
    ]

    if len(dosa_pixels) == 0:
        return 0.0, None

    mean_value = np.mean(
        dosa_pixels
    )

    std_value = np.std(
        dosa_pixels
    )

    # Moderate variation indicates visible browning
    # Too little variation = pale
    # Too much variation = uneven
    target_mean = 150
    target_std = 30

    mean_score = max(
        0,
        100 -
        abs(mean_value - target_mean)
        / target_mean * 100
    )

    variation_score = max(
        0,
        100 -
        abs(std_value - target_std)
        / target_std * 100
    )

    score = (
        mean_score * 0.5
        +
        variation_score * 0.5
    )

    score = max(
        0,
        min(100, score)
    )

    # Create browning heatmap
    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # Only show heatmap inside dosa
    heatmap_masked = np.zeros_like(
        heatmap
    )

    heatmap_masked[
        mask > 0
    ] = heatmap[
        mask > 0
    ]

    return (
        round(score, 1),
        heatmap_masked
    )


# ============================================================
# PORE ANALYSIS
# ============================================================

def calculate_pores(image, mask):
    """
    Detect candidate dark pore-like regions
    inside the dosa.

    This is a visual estimate and may detect
    dark cooking spots as well as actual pores.
    """

    if image is None or mask is None:
        return 0.0, None, 0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur slightly to remove tiny noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Dark-region threshold
    dark_threshold = np.percentile(
        blurred[mask > 0],
        18
    )

    pore_binary = np.zeros_like(
        gray
    )

    pore_binary[
        (blurred < dark_threshold)
        &
        (mask > 0)
    ] = 255

    # Remove very small noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    pore_binary = cv2.morphologyEx(
        pore_binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        pore_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidate_pores = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore extremely tiny noise
        if area < 3:
            continue

        # Ignore very large dark regions
        if area > 500:
            continue

        candidate_pores.append(
            contour
        )

    pore_count = len(
        candidate_pores
    )

    # Score based on a reasonable visible pore count.
    # This is deliberately a visual estimate.
    if pore_count == 0:
        pore_score = 20
    elif pore_count < 5:
        pore_score = 35
    elif pore_count < 15:
        pore_score = 55
    elif pore_count < 30:
        pore_score = 75
    elif pore_count < 60:
        pore_score = 90
    else:
        pore_score = 100

    # Visualization
    pore_visual = image.copy()

    cv2.drawContours(
        pore_visual,
        candidate_pores,
        -1,
        (0, 0, 255),
        2
    )

    return (
        float(pore_score),
        pore_visual,
        pore_count
    )


# ============================================================
# OVERALL DOSA SCORE
# ============================================================

def calculate_overall_score(
    roundness,
    symmetry,
    edge_quality,
    center_accuracy,
    browning,
    pores
):
    """
    Final Dosa DNA score.

    Weights:
        Roundness       25%
        Symmetry        20%
        Edge Quality    15%
        Center Accuracy 15%
        Browning        15%
        Pores           10%
    """

    overall = (
        roundness * 0.25
        +
        symmetry * 0.20
        +
        edge_quality * 0.15
        +
        center_accuracy * 0.15
        +
        browning * 0.15
        +
        pores * 0.10
    )

    return round(
        overall,
        1
    )