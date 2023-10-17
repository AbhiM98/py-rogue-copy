"""Class for bounding boxes."""

from __future__ import annotations


class BBox:
    """Bounding box for a crop of an image."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ):
        """Initialize a BBox."""
        self.x = x  # x coordinate of the top left corner of the crop
        self.y = y  # y coordinate of the top left corner of the crop
        self.width = width  # width of the crop
        self.height = height  # height of the crop

    @property
    def x_max(self) -> int:
        """Return the x coordinate of the bottom right corner of the crop."""
        return self.x + self.width

    @property
    def y_max(self) -> int:
        """Return the y coordinate of the bottom right corner of the crop."""
        return self.y + self.height

    @staticmethod
    def from_list(bbox: list[float]):
        """Return a BBox from a list."""
        return BBox(*bbox)

    def as_list(self) -> list[float]:
        """Return the bounding box as a list."""
        return [self.x, self.y, self.width, self.height]

    def contains_bbox(self, bbox: BBox):
        """Return True if the bbox is completely contained in this bbox."""
        if (
            bbox.x >= self.x
            and bbox.x_max <= self.x_max
            and bbox.y >= self.y
            and bbox.y_max <= self.y_max
        ):
            return True
        else:
            return False

    def partially_contains_bbox(self, bbox: BBox):
        """Return True if the bbox is only partially contained in this bbox."""
        if (
            bbox.x < self.x_max
            and bbox.x_max > self.x
            and bbox.y < self.y_max
            and bbox.y_max > self.y
        ):
            return True
        else:
            return False
