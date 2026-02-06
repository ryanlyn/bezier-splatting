"""Tests for SVG export (svg.py)."""

import re
import xml.etree.ElementTree as ET

import pytest
import torch

from bezier_splatting.coords import model_to_pixel
from bezier_splatting.model import VectorGraphicsScene
from bezier_splatting.svg import _closed_curve_to_path, _open_curve_to_path, _rgb_str, scene_to_svg


# ─── Helpers ──────────────────────────────────────────────────────────


def _parse_svg_path_numbers(d: str) -> list[float]:
    """Extract all numeric values from an SVG path data string."""
    return [float(x) for x in re.findall(r"-?[\d]+\.?\d*", d)]


def _parse_svg_elements(svg_str: str) -> ET.Element:
    """Parse SVG string into ElementTree root."""
    return ET.fromstring(svg_str)


# ─── Color conversion ────────────────────────────────────────────────


class TestRgbStr:
    def test_black(self):
        assert _rgb_str(torch.tensor([0.0, 0.0, 0.0])) == "rgb(0,0,0)"

    def test_white(self):
        assert _rgb_str(torch.tensor([1.0, 1.0, 1.0])) == "rgb(255,255,255)"

    def test_clamps_above_1(self):
        result = _rgb_str(torch.tensor([2.0, 0.5, -0.5]))
        assert result == "rgb(255,127,0)"


# ─── Open curve path construction ────────────────────────────────────


class TestOpenCurvePath:
    """Verify open curve SVG path construction with 3 cubic segments."""

    @pytest.fixture()
    def open_curve_data(self):
        """Open curve with known CPs in [-1, 1]."""
        # 10 control points, linearly spaced for easy verification
        cp = torch.zeros(10, 2)
        for i in range(10):
            cp[i, 0] = -1.0 + i * 2.0 / 9  # x: [-1, 1]
            cp[i, 1] = 0.0  # y: all zero (horizontal line)
        color = torch.zeros(3)  # pre-sigmoid -> sigmoid(0) = 0.5
        opacity = 0.8  # post-sigmoid
        stroke_width = 0.0  # pre-sigmoid -> 0.5 + 0.5 * 4.5 = 2.75
        return cp, color, opacity, stroke_width

    def test_path_has_three_cubic_segments(self, open_curve_data):
        cp, color, opacity, sw = open_curve_data
        path_str = _open_curve_to_path(cp, color, opacity, sw, 64, 64)
        # Should have M + 3 C commands
        assert path_str.count(" C ") == 3
        assert "M " in path_str

    def test_coordinate_round_trip(self, open_curve_data):
        """Verify SVG pixel coords match model_to_pixel conversion."""
        cp, color, opacity, sw = open_curve_data
        H, W = 128, 256
        path_str = _open_curve_to_path(cp, color, opacity, sw, H, W)

        expected_px = model_to_pixel(cp, H, W)
        # Extract just the path 'd' attribute
        d_match = re.search(r'd="([^"]+)"', path_str)
        assert d_match is not None
        numbers = _parse_svg_path_numbers(d_match.group(1))

        # Should be 10 points * 2 coords = 20 numbers
        assert len(numbers) == 20

        for i in range(10):
            assert numbers[2 * i] == pytest.approx(expected_px[i, 0].item(), abs=0.01)
            assert numbers[2 * i + 1] == pytest.approx(expected_px[i, 1].item(), abs=0.01)

    def test_segment_cp_indices(self, open_curve_data):
        """Verify CP indices: seg0=[0:4], seg1=[3:7], seg2=[6:10]."""
        cp, color, opacity, sw = open_curve_data
        H, W = 64, 64
        path_str = _open_curve_to_path(cp, color, opacity, sw, H, W)
        expected_px = model_to_pixel(cp, H, W)

        d_match = re.search(r'd="([^"]+)"', path_str)
        numbers = _parse_svg_path_numbers(d_match.group(1))
        coords = [(numbers[2 * i], numbers[2 * i + 1]) for i in range(10)]

        # Shared points: index 3 appears as endpoint of seg0 and start of seg1
        # In the path: M cp[0] C cp[1] cp[2] cp[3] C cp[4] cp[5] cp[6] C cp[7] cp[8] cp[9]
        # cp[3] is the 4th point (index 3), cp[6] is the 7th (index 6)
        px3 = expected_px[3].tolist()
        px6 = expected_px[6].tolist()
        assert coords[3] == pytest.approx(px3, abs=0.01)
        assert coords[6] == pytest.approx(px6, abs=0.01)

    def test_stroke_width_sigmoid(self, open_curve_data):
        """Stroke width: 0.5 + sigmoid(w) * 4.5."""
        cp, color, opacity, sw = open_curve_data
        path_str = _open_curve_to_path(cp, color, opacity, sw, 64, 64)
        expected_sw = 0.5 + torch.sigmoid(torch.tensor(sw)).item() * 4.5
        sw_match = re.search(r'stroke-width="([^"]+)"', path_str)
        assert sw_match is not None
        assert float(sw_match.group(1)) == pytest.approx(expected_sw, abs=0.01)

    def test_color_sigmoid_applied(self, open_curve_data):
        """Color: sigmoid(pre_sigmoid) -> rgb string."""
        cp, color, opacity, sw = open_curve_data
        path_str = _open_curve_to_path(cp, color, opacity, sw, 64, 64)
        # color = [0, 0, 0] -> sigmoid = [0.5, 0.5, 0.5] -> rgb(127, 127, 127)
        assert 'stroke="rgb(127,127,127)"' in path_str

    def test_opacity_in_output(self, open_curve_data):
        cp, color, opacity, sw = open_curve_data
        path_str = _open_curve_to_path(cp, color, opacity, sw, 64, 64)
        assert 'opacity="0.800"' in path_str

    def test_fill_none(self, open_curve_data):
        cp, color, opacity, sw = open_curve_data
        path_str = _open_curve_to_path(cp, color, opacity, sw, 64, 64)
        assert 'fill="none"' in path_str


# ─── Closed curve path construction ──────────────────────────────────


class TestClosedCurvePath:
    """Verify closed curve SVG path construction."""

    @pytest.fixture()
    def closed_curve_data(self):
        """Closed curve with 2 boundaries, 4 CPs each, in [-1, 1]."""
        boundary_cp = torch.tensor([
            # Top boundary: arc above y=0
            [[-0.5, 0.0], [-0.25, 0.5], [0.25, 0.5], [0.5, 0.0]],
            # Bottom boundary: arc below y=0
            [[-0.5, 0.0], [-0.25, -0.5], [0.25, -0.5], [0.5, 0.0]],
        ])
        color = torch.tensor([2.0, -2.0, 0.0])  # pre-sigmoid
        opacity = 0.9  # post-sigmoid
        return boundary_cp, color, opacity

    def test_path_closes_with_z(self, closed_curve_data):
        boundary_cp, color, opacity = closed_curve_data
        path_str = _closed_curve_to_path(boundary_cp, color, opacity, 64, 64)
        d_match = re.search(r'd="([^"]+)"', path_str)
        assert d_match is not None
        assert d_match.group(1).strip().endswith("Z")

    def test_has_fill_color(self, closed_curve_data):
        boundary_cp, color, opacity = closed_curve_data
        path_str = _closed_curve_to_path(boundary_cp, color, opacity, 64, 64)
        # sigmoid([2, -2, 0]) -> [0.88, 0.12, 0.5] -> rgb(224, 30, 127)
        assert 'fill="rgb(' in path_str
        assert 'opacity="0.900"' in path_str

    def test_coordinate_round_trip(self, closed_curve_data):
        """Pixel coordinates in SVG match model_to_pixel."""
        boundary_cp, color, opacity = closed_curve_data
        H, W = 128, 256
        path_str = _closed_curve_to_path(boundary_cp, color, opacity, H, W)

        expected_top = model_to_pixel(boundary_cp[0], H, W)
        expected_bot = model_to_pixel(boundary_cp[1], H, W)

        d_match = re.search(r'd="([^"]+)"', path_str)
        numbers = _parse_svg_path_numbers(d_match.group(1))

        # Path: M top[0] C top[1] top[2] top[3] L bot[3] C bot[2] bot[1] bot[0] Z
        # = 4 top points + 1 line-to (bot end) + 3 reversed bot points = 8 points = 16 numbers
        assert len(numbers) == 16

        # First 4 points are top boundary
        for i in range(4):
            assert numbers[2 * i] == pytest.approx(expected_top[i, 0].item(), abs=0.01)
            assert numbers[2 * i + 1] == pytest.approx(expected_top[i, 1].item(), abs=0.01)

        # Next is bot[-1] (line-to)
        assert numbers[8] == pytest.approx(expected_bot[3, 0].item(), abs=0.01)
        assert numbers[9] == pytest.approx(expected_bot[3, 1].item(), abs=0.01)

        # Then reversed bot: [2], [1], [0]
        assert numbers[10] == pytest.approx(expected_bot[2, 0].item(), abs=0.01)
        assert numbers[11] == pytest.approx(expected_bot[2, 1].item(), abs=0.01)
        assert numbers[12] == pytest.approx(expected_bot[1, 0].item(), abs=0.01)
        assert numbers[13] == pytest.approx(expected_bot[1, 1].item(), abs=0.01)
        assert numbers[14] == pytest.approx(expected_bot[0, 0].item(), abs=0.01)
        assert numbers[15] == pytest.approx(expected_bot[0, 1].item(), abs=0.01)

    def test_shared_endpoints(self, closed_curve_data):
        """Shared endpoints: top[0]==bot[0] and top[-1]==bot[-1]."""
        boundary_cp, color, opacity = closed_curve_data
        # Both boundaries start at [-0.5, 0] and end at [0.5, 0]
        assert torch.allclose(boundary_cp[0, 0], boundary_cp[1, 0])
        assert torch.allclose(boundary_cp[0, -1], boundary_cp[1, -1])

    def test_cubic_used_for_4cp(self, closed_curve_data):
        """4-CP boundaries use cubic Bezier (C) not line segments (L)."""
        boundary_cp, color, opacity = closed_curve_data
        path_str = _closed_curve_to_path(boundary_cp, color, opacity, 64, 64)
        d_match = re.search(r'd="([^"]+)"', path_str)
        d = d_match.group(1)
        # Should have 2 C commands (top forward + bottom reversed) + 1 L (connector)
        assert d.count(" C ") == 2
        assert d.count(" L ") == 1  # connector from top end to bot end


# ─── scene_to_svg (full export) ──────────────────────────────────────


class TestSceneToSvg:
    """Full scene export tests."""

    def test_empty_scene(self):
        """Empty scene produces valid SVG with just background rect."""
        scene = VectorGraphicsScene(n_open=0, n_closed=0, H=64, W=64)
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        assert root.tag.endswith("svg")
        assert root.attrib["width"] == "64"
        assert root.attrib["height"] == "64"
        # Should have just the background rect
        rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
        assert len(rects) == 1

    def test_open_only_scene(self):
        """Scene with open curves only produces valid SVG."""
        scene = VectorGraphicsScene(n_open=3, n_closed=0, H=64, W=64)
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns)
        # May be fewer than 3 if some have opacity < 0.01
        # With init opacity=0 -> sigmoid(0)=0.5 -> mean=0.5 > 0.01, so all should appear
        assert len(paths) == 3

    def test_closed_only_scene(self):
        """Scene with closed curves only produces valid SVG."""
        scene = VectorGraphicsScene(n_open=0, n_closed=2, H=64, W=64)
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns)
        assert len(paths) == 2

    def test_depth_order_larger_area_first(self):
        """Larger area curves appear before smaller ones in SVG (painter's algorithm)."""
        torch.manual_seed(42)
        scene = VectorGraphicsScene(n_open=2, n_closed=0, H=64, W=64)

        # Make curve 0 much bigger than curve 1
        with torch.no_grad():
            # Curve 0: large spread
            cp0 = torch.linspace(-0.9, 0.9, 10).unsqueeze(-1).expand(-1, 2).clone()
            cp0[:, 1] = 0.0
            scene.open_control_points[0] = cp0

            # Curve 1: tiny
            cp1 = torch.linspace(-0.05, 0.05, 10).unsqueeze(-1).expand(-1, 2).clone()
            cp1[:, 1] = 0.0
            scene.open_control_points[1] = cp1

            # Both visible
            scene.open_opacities.fill_(2.0)  # sigmoid(2) ≈ 0.88

        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns)

        assert len(paths) == 2

        # First path should have wider coordinate spread (larger area)
        d0 = paths[0].attrib["d"]
        d1 = paths[1].attrib["d"]
        nums0 = _parse_svg_path_numbers(d0)
        nums1 = _parse_svg_path_numbers(d1)
        xs0 = nums0[::2]  # x coords
        xs1 = nums1[::2]
        spread0 = max(xs0) - min(xs0)
        spread1 = max(xs1) - min(xs1)

        assert spread0 > spread1, "Larger curve should be drawn first (background)"

    def test_color_accuracy(self):
        """SVG fill/stroke colors match sigmoid(pre_sigmoid) * 255."""
        scene = VectorGraphicsScene(n_open=1, n_closed=0, H=64, W=64)
        with torch.no_grad():
            scene.open_colors[0] = torch.tensor([2.0, -2.0, 0.0])
            scene.open_opacities[0].fill_(2.0)  # ensure visible
        svg = scene_to_svg(scene)

        # sigmoid([2, -2, 0]) -> [0.8808, 0.1192, 0.5] -> int([224, 30, 127])
        expected_r = int(torch.sigmoid(torch.tensor(2.0)).item() * 255)
        expected_g = int(torch.sigmoid(torch.tensor(-2.0)).item() * 255)
        expected_b = int(torch.sigmoid(torch.tensor(0.0)).item() * 255)
        expected_rgb = f"rgb({expected_r},{expected_g},{expected_b})"
        assert expected_rgb in svg

    def test_opacity_open_is_mean_sigmoid(self):
        """Open curve SVG opacity = mean(sigmoid(per-segment opacities))."""
        scene = VectorGraphicsScene(n_open=1, n_closed=0, H=64, W=64)
        with torch.no_grad():
            scene.open_opacities[0] = torch.tensor([1.0, 2.0, 3.0])
        svg = scene_to_svg(scene)

        expected = torch.sigmoid(torch.tensor([1.0, 2.0, 3.0])).mean().item()
        assert f'opacity="{expected:.3f}"' in svg

    def test_opacity_closed_is_sigmoid(self):
        """Closed curve SVG opacity = sigmoid(single opacity)."""
        scene = VectorGraphicsScene(n_open=0, n_closed=1, H=64, W=64)
        with torch.no_grad():
            scene.closed_opacities[0] = 1.5
        svg = scene_to_svg(scene)

        expected = torch.sigmoid(torch.tensor(1.5)).item()
        assert f'opacity="{expected:.3f}"' in svg

    def test_svg_dimensions(self):
        """SVG dimensions match H, W arguments."""
        scene = VectorGraphicsScene(n_open=0, n_closed=0, H=100, W=200)
        svg = scene_to_svg(scene, H=300, W=400)
        root = _parse_svg_elements(svg)
        assert root.attrib["width"] == "400"
        assert root.attrib["height"] == "300"
        assert root.attrib["viewBox"] == "0 0 400 300"

    def test_default_dimensions_from_scene(self):
        """SVG defaults to scene.H, scene.W when not specified."""
        scene = VectorGraphicsScene(n_open=0, n_closed=0, H=128, W=256)
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        assert root.attrib["width"] == "256"
        assert root.attrib["height"] == "128"

    def test_mixed_scene(self):
        """Scene with both open and closed curves exports correctly."""
        scene = VectorGraphicsScene(n_open=2, n_closed=1, H=64, W=64)
        with torch.no_grad():
            scene.open_opacities.fill_(2.0)
            scene.closed_opacities.fill_(2.0)
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns)
        # 2 open + 1 closed = 3 paths
        assert len(paths) == 3

    def test_low_opacity_curves_filtered(self):
        """Curves with sigmoid(opacity) < 0.01 are excluded from SVG."""
        scene = VectorGraphicsScene(n_open=2, n_closed=0, H=64, W=64)
        with torch.no_grad():
            scene.open_opacities[0].fill_(2.0)  # visible
            scene.open_opacities[1].fill_(-10.0)  # sigmoid(-10) ≈ 0.00005 < 0.01
        svg = scene_to_svg(scene)
        root = _parse_svg_elements(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns)
        assert len(paths) == 1
