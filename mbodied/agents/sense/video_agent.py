import argparse
import json
import sys
import time
from typing import NoReturn

import cv2
import imutils
import zenoh
from imutils.video import VideoStream
from lager import log

from mbodied.types.sample import Sample

log.add(sys.stdout, colorize=True, level="INFO")
log.info(globals().keys())
CAMERA_ID = 0

parser = argparse.ArgumentParser(prog="zcapture", description="zenoh video capture example")
parser.add_argument("-m", "--mode", type=str, choices=["peer", "client"], help="The zenoh session mode.")
parser.add_argument(
    "-e", "--connect", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on."
)
parser.add_argument(
    "-l", "--listen", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on."
)
parser.add_argument("-w", "--width", type=int, default=500, help="width of the published frames")
parser.add_argument("-q", "--quality", type=int, default=95, help="quality of the published frames (0 - 100)")
parser.add_argument("-d", "--delay", type=float, default=0.05, help="delay between each frame in seconds")
parser.add_argument("-k", "--key", type=str, default="demo/zcam", help="key expression")
parser.add_argument("-c", "--config", type=str, metavar="FILE", help="A zenoh configuration file.")



def main(args) -> NoReturn:
    conf: zenoh.Config = zenoh.Config.from_json5(config_str)
    if args.mode is not None:
        conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(args.mode))
    if args.connect is not None:
        conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
    if args.listen is not None:
        conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(args.listen))
    log.info("[INFO] Open zenoh session...")
    zenoh.init_logger()
    z = zenoh.open(conf)

    log.info("[INFO] Open camera...")
    vs = VideoStream(src=CAMERA_ID).start()

    time.sleep(1.0)

    while True:
        raw = vs.read()
        if raw is not None:
            frame = imutils.resize(raw, width=args.width)
            _, jpeg = cv2.imencode(".jpg", frame, jpeg_opts)
            z.put(args.key, jpeg.tobytes())

        time.sleep(args.delay)

if __name__ == "__main__":
  args = parser.parse_args()

  jpeg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]

  config_str ="""
  {
    
      storage_manager: {             // activate and configure the storage_manager plugin
        storages: {
          myhome: {                  // configure a "myhome" storage
            key_expr: "demo/zcam",   // which subscribes and replies to query on myhome/**
            volume: {                // and using the "memory" volume (always present by default)
              id: "memory"
            }
          }
        },
      },
      connect: {
        /// timeout waiting for all endpoints connected (0: no retry, -1: infinite timeout)
        /// Accepts a single value or different values for router, peer and client.
        timeout_ms: { router: -1, peer: -1, client: 0 },

        endpoints: [
          "udp/3.236.52.5:7861",
        ],

        /// Global connect configuration,
        /// Accepts a single value or different values for router, peer and client.
        /// The configuration can also be specified for the separate endpoint
        /// it will override the global one
        /// E.g. tcp/192.168.0.1:7447#retry_period_init_ms=20000;retry_period_max_ms=10000"

        /// exit from application, if timeout exceed
        exit_on_failure: { router: false, peer: false, client: true },
        /// connect establishing retry configuration
        retry: {
          /// initial wait timeout until next connect try
          period_init_ms: 1000,
          /// maximum wait timeout until next connect try
          period_max_ms: 4000,
          /// increase factor for the next timeout until nexti connect try
          period_increase_factor: 2,
        },
      },
  }
  """
