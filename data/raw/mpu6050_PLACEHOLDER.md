# MPU6050 — PLACEHOLDER SUMMARY

> **PLACEHOLDER.** This is a short hand-written summary used only to test the
> pipeline end to end. It is NOT the official InvenSense/TDK datasheet. Replace
> it with the real datasheet PDF before evaluation.

## Overview

The MPU6050 is a 6-axis motion tracking device that combines a 3-axis gyroscope
and a 3-axis accelerometer on a single chip, together with an onboard Digital
Motion Processor (DMP). It is commonly used as an IMU for balancing robots,
drones, and orientation estimation.

## Gyroscope

The gyroscope measures angular velocity. Its full-scale range is user
programmable to ±250, ±500, ±1000, or ±2000 degrees per second. Each axis is
digitised by a 16-bit ADC.

## Accelerometer

The accelerometer measures linear acceleration. Its full-scale range is user
programmable to ±2g, ±4g, ±8g, or ±16g, also with 16-bit ADCs.

## Communication

The MPU6050 talks to a host microcontroller over I2C at up to 400 kHz.
Its 7-bit I2C address is 0x68 when the AD0 pin is low and 0x69 when AD0 is
high, which allows two sensors on the same bus. The WHO_AM_I register (0x75)
returns 0x68 and is a quick way to confirm the device is wired correctly.

## Other features

- 1024-byte FIFO buffer to reduce host polling.
- On-chip temperature sensor.
- Supply voltage (VDD) range of 2.375 V to 3.46 V; breakout boards usually add
  a regulator so they can be powered from 5 V.
