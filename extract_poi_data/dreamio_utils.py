from shapely.geometry import Polygon, MultiPolygon
from typing import Optional


def polygon_to_list(geom) -> Optional[list[list[tuple[float, float]]]]:
    """
    Converts a Shapely Polygon or MultiPolygon object to a list of coordinate lists.

    Args:
        geom: A Shapely Polygon or MultiPolygon object.

    Returns:
        A list of lists of coordinates, where each coordinate is a tuple of (x, y) floats.
        For Polygon: Returns [[coords]] - single list of coordinates
        For MultiPolygon: Returns [[coords1], [coords2], ...] - list of coordinate lists
    """
    if geom is None:
        return None

    try:
        if geom.geom_type == "Polygon":
            # Extract exterior coordinates and remove the duplicate last point
            coords = list(geom.exterior.coords)[:-1]
            return [coords]
        elif geom.geom_type == "MultiPolygon":
            # Handle each polygon in the multipolygon
            return [list(polygon.exterior.coords)[:-1] for polygon in geom.geoms]
        else:
            return None
    except Exception as e:
        print(f"Error processing geometry: {e}")
        return None


def list_to_polygon(coord_lists: list[list[tuple[float, float]]]):
    """
    Converts a list of coordinate lists into a Shapely Polygon or MultiPolygon.

    Args:
        coord_lists: A list of lists of coordinates. Each inner list represents a polygon's exterior coordinates.

    Returns:
        A Shapely Polygon if there's one list, or a MultiPolygon if there are multiple lists.
    """
    if len(coord_lists) == 1:
        return Polygon(coord_lists[0])
    elif len(coord_lists) > 1:
        return MultiPolygon([Polygon(coords) for coords in coord_lists])
    else:
        raise ValueError("Coordinate list cannot be empty.")


def main():
    # Example Polygons
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    multipolygon = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
        ]
    )

    # Convert to List
    print("Polygon to List:")
    polygon_list = polygon_to_list(polygon)
    print(polygon_list)

    print("\nMultiPolygon to List:")
    multipolygon_list = polygon_to_list(multipolygon)
    print(multipolygon_list)

    # Convert Back to Geometry
    print("\nList to Polygon:")
    converted_polygon = list_to_polygon(polygon_list)
    print(converted_polygon)

    print("\nList to MultiPolygon:")
    converted_multipolygon = list_to_polygon(multipolygon_list)
    print(converted_multipolygon)

    # Test Cases
    assert polygon_to_list(polygon) == [
        [(0, 0), (1, 0), (1, 1), (0, 1)]
    ], polygon_to_list(polygon)
    assert polygon_to_list(multipolygon) == [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(2, 2), (3, 2), (3, 3), (2, 3)],
    ]

    print("\nTests passed successfully!")


if __name__ == "__main__":
    main()
