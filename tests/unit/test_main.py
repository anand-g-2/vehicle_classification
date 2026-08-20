"""
Unit tests for main.py's CLI parsing — pure argparse logic, no video/model
involved.
"""
from __future__ import annotations

import pytest

from main import DEFAULT_MODEL, DEFAULT_VIDEO, parse_args


def test_defaults():
    args = parse_args([])

    assert args.video == str(DEFAULT_VIDEO)
    assert args.model == str(DEFAULT_MODEL)
    assert args.tracker == "ByteTrack"
    assert args.output == "vehicle_track.mp4"
    assert args.intersection_threshold == 50.0
    assert args.no_display is False


def test_overrides():
    args = parse_args([
        "--video", "clip.mp4",
        "--model", "weights/custom.pt",
        "--tracker", "BotSort",
        "--output", "out.mp4",
        "--intersection-threshold", "75",
        "--no-display",
    ])

    assert args.video == "clip.mp4"
    assert args.model == "weights/custom.pt"
    assert args.tracker == "BotSort"
    assert args.output == "out.mp4"
    assert args.intersection_threshold == 75.0
    assert args.no_display is True


def test_invalid_tracker_choice_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--tracker", "NotARealTracker"])
