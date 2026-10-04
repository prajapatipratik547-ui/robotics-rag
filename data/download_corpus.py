"""Download the corpus into data/raw/.

Datasheets are the manufacturers' copyrighted documents, so they are not
committed to git. This script records exactly where each one comes from,
so anyone can rebuild the same corpus:

  python data/download_corpus.py           # skip files already downloaded
  python data/download_corpus.py --force   # re-download everything

File names matter: ingestion prepends the file name to every chunk, so a
name like 'mpu6050_register_map.pdf' tells retrieval which part a page is about.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# (file name in data/raw, URL). Official manufacturer URLs where they are
# downloadable; otherwise the manufacturer's own PDF as hosted by a
# distributor or university (noted per line).
DATASHEETS = [
    # Slamtec's official English datasheet v3.0 (LD108). slamtec.com returns 403 to
    # scripted downloads, so this is the same PDF from DFRobot's GitHub. (Farnell's
    # "datasheet" for this part is a printed product page with the specs as images.)
    ("rplidar_a1m8_datasheet.pdf",
     "https://github.com/May-DFRobot/DFRobot/raw/master/LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf"),
    ("raspberry_pi_4b_datasheet.pdf", "https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf"),
    ("arduino_mega_2560_datasheet.pdf", "https://docs.arduino.cc/resources/datasheets/A000067-datasheet.pdf"),
    # InvenSense (TDK) documents PS-MPU-6000A-00 and RM-MPU-6000A-00, hosted by SparkFun.
    ("mpu6050_product_specification.pdf",
     "https://cdn.sparkfun.com/datasheets/Components/General%20IC/PS-MPU-6000A.pdf"),
    ("mpu6050_register_map.pdf", "https://cdn.sparkfun.com/datasheets/Sensors/Accelerometers/RM-MPU-6000A.pdf"),
    ("neo-6m_gps_datasheet.pdf",
     "https://content.u-blox.com/sites/default/files/products/documents/NEO-6_DataSheet_%28GPS.G6-HW-09005%29.pdf"),
    # ST's site blocks scripted downloads; this is ST's L298 datasheet hosted by Electrokit.
    ("l298n_motor_driver_datasheet.pdf", "https://www.electrokit.com/upload/quick/4f/70/f1f1_40350298-tds.pdf"),
    # HC-SR04 and MG996R have no manufacturer download; these are the widely used vendor sheets.
    ("hc-sr04_ultrasonic_sensor_datasheet.pdf", "https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf"),
    ("mg996r_servo_datasheet.pdf",
     "https://courses.ece.cornell.edu/ece5990/ECE5725_Spring2020_Projects/May_15_Demo/Smart%20Lock/"
     "W_pp445_yl656/datasheet/MG996R_Tower-Pro.pdf"),
]

# Nav2 documentation for ROS 2 Jazzy (the LTS release that runs on Ubuntu
# 24.04, i.e. what a Raspberry Pi 4 robot would typically use).
NAV2_BASE = "https://docs.nav2.org/jazzy/"
NAV2_PAGES = [
    ("nav2_concepts_ros2.html", "getting_started/navigation_concepts/ros2/"),
    ("nav2_concepts_behavior_trees.html", "getting_started/navigation_concepts/behavior_trees/"),
    ("nav2_concepts_navigation_servers.html", "getting_started/navigation_concepts/navigation_servers/"),
    ("nav2_concepts_state_estimation.html", "getting_started/navigation_concepts/state_estimation/"),
    ("nav2_concepts_environmental_representation.html",
     "getting_started/navigation_concepts/environmental_representation/"),
    ("nav2_setup_sensors.html",
     "configuration_and_development/first_time_robot_setup_guide/sensors/setup_sensors_gz/"),
    ("nav2_setup_odometry.html", "configuration_and_development/first_time_robot_setup_guide/odom/setup_odom_gz/"),
    ("nav2_setup_transforms.html",
     "configuration_and_development/first_time_robot_setup_guide/transformation/setup_transforms/"),
    ("nav2_controller_server.html", "configuration_and_development/configuration_guide/core_servers/controller_server/"),
    ("nav2_planner_server.html",
     "configuration_and_development/configuration_guide/core_servers/configuring_planner_server/"),
    ("nav2_behavior_server.html",
     "configuration_and_development/configuration_guide/core_servers/configuring_behavior_server/"),
    ("nav2_bt_navigator.html", "configuration_and_development/configuration_guide/core_servers/configuring_bt_navigator/"),
    ("nav2_costmap_2d.html", "configuration_and_development/configuration_guide/core_servers/costmap_2d/"),
    ("nav2_costmap_inflation_layer.html",
     "configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/inflation/"),
    ("nav2_costmap_obstacle_layer.html",
     "configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/obstacle/"),
    ("nav2_amcl.html", "configuration_and_development/configuration_guide/others/configuring_amcl/"),
    ("nav2_regulated_pure_pursuit_controller.html",
     "configuration_and_development/configuration_guide/controller_plugins/configuring_regulated_pp/"),
    ("nav2_mppi_controller.html",
     "configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/"),
    ("nav2_navfn_planner.html", "configuration_and_development/configuration_guide/planners_plugins/configuring_navfn/"),
    ("nav2_smac_hybrid_planner.html",
     "configuration_and_development/configuration_guide/planners_plugins/smac/smac_hybrid/configuring_smac_hybrid/"),
    ("nav2_tuning_guide.html", "configuration_and_development/tuning_guide/"),
    ("nav2_with_slam_tutorial.html", "tutorials/general_tutorials/navigation2_with_slam/navigation2_with_slam/"),
    ("nav2_with_gps_tutorial.html", "tutorials/general_tutorials/navigation2_with_gps/navigation2_with_gps/"),
]

SOURCES = DATASHEETS + [(name, NAV2_BASE + path) for name, path in NAV2_PAGES]
# Some distributor sites (Farnell) reset connections for non-browser user agents.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def looks_valid(name: str, data: bytes) -> bool:
    """Catch the common failure where a server returns an HTML error or
    'page moved' page instead of the real document."""
    if name.endswith(".pdf"):
        return data.startswith(b"%PDF")
    return b"<html" in data[:2000].lower() and b"This page moved" not in data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, url in SOURCES:
        target = config.RAW_DATA_DIR / name
        if target.exists() and not args.force:
            print(f"  exists   {name}")
            continue
        try:
            data = fetch(url)
        except Exception as e:  # network errors, 404s, timeouts
            print(f"  FAILED   {name}: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        if not looks_valid(name, data):
            print(f"  INVALID  {name}: server did not return the expected document")
            failed.append(name)
            continue
        target.write_bytes(data)
        print(f"  saved    {name} ({len(data) // 1024} KB)")
        time.sleep(0.5)  # be polite to the documentation servers

    print(f"\n{len(SOURCES) - len(failed)}/{len(SOURCES)} documents in {config.RAW_DATA_DIR}")
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
