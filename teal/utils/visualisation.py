import numpy as np
import cv2


def draw_text(
    image: np.ndarray,
    text,
    center,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    fontsize: float = 1.0,
    thickness=1,
    color=(255, 255, 255),
):
    textsize = cv2.getTextSize(text, font, fontsize, thickness)[0]
    text_x = center[0] - textsize[0] // 2
    text_y = center[1] + textsize[1] // 2
    cv2.putText(
        image, text, (text_x, text_y), font, fontsize, color, thickness, cv2.LINE_AA
    )


def video_bar_chart(values: np.ndarray, width: int, height: int, bar_color=(0, 0, 255)):
    """
    Returns a bar chat of a time series. The output
    has shape (T, H, W, 3). All time steps share the
    same bars and the current bar is highlighted with
    a white vertical line.
    """
    assert values.ndim == 1, f"Expected values to have 1 dimension, got {values.ndim}"
    assert values.shape[0] > 1, f"Expected values to have at least 2 elements"
    assert (
        values.dtype == np.float32
    ), f"Expected values to have dtype np.float32, got {values.dtype}"
    assert width > 0, f"Expected width to be positive, got {width}"
    assert height > 0, f"Expected height to be positive, got {height}"
    has_neg_values = np.any(values < 0)
    max_value = max(np.max(np.abs(values)), 1.0)
    half_height = height // 2

    def value_to_height(v):
        if has_neg_values:
            # if negative values are present, split images in half
            # where the bottom half is for negative values and the
            # top half is for positive values
            return int(half_height - v / max_value * half_height)
        else:
            return int(height - v / max_value * height)

    # draw bars
    bar_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = width // values.shape[0]
    assert bar_width > 0, f"Expected bar_width to be positive, got {bar_width}"
    for i, v in enumerate(values):
        cv2.rectangle(
            bar_canvas,
            (i * bar_width, value_to_height(0)),
            ((i + 1) * bar_width, value_to_height(v)),
            bar_color,
            -1,
        )

    # draw 0 horizontal line
    cv2.line(
        bar_canvas,
        (0, value_to_height(0)),
        (width, value_to_height(0)),
        (200, 200, 200),
        1,
    )

    # draw vertical lines
    canvas = np.broadcast_to(bar_canvas, (values.shape[0], height, width, 3)).copy()
    for i in range(values.shape[0]):
        # draw line centered at the current bar
        cv2.line(
            canvas[i],
            (i * bar_width + bar_width // 2, 0),
            (i * bar_width + bar_width // 2, height),
            (255, 255, 255),
            1,
        )
    return canvas
