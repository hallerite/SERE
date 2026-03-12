"""Parse RML-style XML waypoint plans."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_rml_plan(xml_text: str) -> List[Tuple[float, float]]:
    """Parse <rover-plan> XML into a list of (x, y) waypoints."""
    xml_text = xml_text.strip()
    if not xml_text:
        raise ValueError("Empty plan")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}") from e

    if root.tag != "rover-plan":
        raise ValueError(f"Expected <rover-plan> root, got <{root.tag}>")

    waypoints: List[Tuple[float, float]] = []
    for elem in root:
        if elem.tag == "waypoint":
            x_s, y_s = elem.get("x"), elem.get("y")
            if x_s is None or y_s is None:
                raise ValueError(
                    f"Waypoint missing x or y: {ET.tostring(elem, encoding='unicode')}"
                )
            try:
                waypoints.append((float(x_s), float(y_s)))
            except ValueError:
                raise ValueError(f"Non-numeric coordinates: x={x_s!r}, y={y_s!r}")
    return waypoints
