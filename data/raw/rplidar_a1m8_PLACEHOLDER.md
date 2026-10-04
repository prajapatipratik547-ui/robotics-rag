# RPLiDAR A1M8 — PLACEHOLDER SUMMARY

> **PLACEHOLDER.** This is a short hand-written summary used only to test the
> pipeline end to end. It is NOT the official Slamtec datasheet. Replace it with
> the real datasheet PDF before evaluation.

## Overview

The RPLiDAR A1M8 is a low-cost 360-degree 2D laser range scanner made by
Slamtec. It is widely used on hobby and research robots for SLAM (simultaneous
localization and mapping) and obstacle avoidance. The scanner spins a laser
range-finding core on a motor and reports distance and angle samples over a
serial link.

## Measurement performance

- Measuring principle: laser triangulation.
- Distance range: 0.15 m to 12 m.
- Angular range: 360 degrees.
- Sample rate: up to 8000 samples per second.
- Scan frequency: typically 5.5 Hz, adjustable from about 1 Hz to 10 Hz by
  changing the motor speed.
- Angular resolution: 1 degree or better at the typical scan frequency.

## Interface

The A1M8 communicates over a 3.3 V TTL UART at 115200 baud. A USB adapter board
ships with the development kit so it can be connected directly to a PC or a
Raspberry Pi. The scan motor is powered from 5 V and its speed can be controlled
with a PWM signal on the MOTOCTL pin.

## Use with ROS 2

Slamtec provides the `rplidar_ros` driver package, which publishes scans as
`sensor_msgs/LaserScan` messages on the `/scan` topic. These scans can be
consumed directly by SLAM Toolbox and by Nav2 costmaps.
