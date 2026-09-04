"""Server and device configuration, shared by every suite.

No device or server is named here: both come from the environment, so the
committed defaults are placeholders that will not resolve against a server.
"""

import os

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 62123

# Set EQ1_SERVER_URL to run against an already-running server; conftest then
# starts none of its own.
SERVER_URL = os.environ.get("EQ1_SERVER_URL", f"http://{SERVER_HOST}:{SERVER_PORT}")
CLIENT_TIMEOUT_S = 60 * 60

# Report label -> backend id. The label is what the report shows.
DEVICES = {
    "Device-Ref": os.environ.get("EQ1_DEVICE_REF", "device-ref"),
    "Device-1": os.environ.get("EQ1_DEVICE_1", "device-1"),
    "Device-2": os.environ.get("EQ1_DEVICE_2", "device-2"),
}
