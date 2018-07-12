"""
JSON Exporter.

Will export JSON to files or K/V store for microservices
to pick up and use. Used to generate the JSON data used by the EGS
v0 API, and similar legacy services.
"""

import json
import os
import numpy as np
from hexbytes import HexBytes
from urllib.parse import urlparse

import redis
from .settings import settings_file_loaded, get_setting

class JSONExporter(object):
    """JSON exporter main class. Allows for export of JSON strings to various
    backing data stores."""

    redis_key_prefix = ""
    supported_types = ['file', 'redis']

    redis = None

    export_type = None
    export_location = None


    def __init__(self, export_type=None, export_location=None):
        if export_type is None:
            self._get_export_options_from_settings()
        else:
            self._check_export_type(export_type)
            self.export_type = export_type
            self.export_location = export_location

    def write_json(self, key, object_or_str):
        """Writes JSON to supported endpoint."""
        if self.export_type == 'file':
            self._write_json_file(key, object_or_str)
        elif self.export_type == 'redis':
            self._write_json_redis(key, object_or_str)

    def _write_json_file(self, key, object_or_str):
        """Writes JSON to filesystem."""
        if not os.path.isdir(self.export_location):
            raise JSONExporterException(
                "Cannot write to output dir %s, doesn't exist." %
                (self.export_location))

        json_str = self._serialize(object_or_str)
        output_path = os.path.join(self.export_location, "%s.json" % (key))
        with open(output_path, 'w') as fd:
            fd.write(json_str)

    def _write_json_redis(self, key, object_or_str):
        """Writes JSON to Redis store."""
        # self.export_location should be parseable
        conn = self._connect_redis()
        key = "%s_%s" % (self.redis_key_prefix, key)
        json_str = self._serialize(object_or_str)
        conn.set(key, json_str)

    def _check_export_type(self, export_type):
        """Checks for a valid export type. Raises Error if not found."""
        if not export_type in self.supported_types:
            raise JSONExporterException("JSONExporter does not support type %s" % export_type)

    def _serialize(self, mixed):
        """Serializes mixed to JSON."""
        if isinstance(mixed, str):
            # serialize to validate is JSON
            mixed = json.loads(mixed)
        elif isinstance(mixed, dict):
            # web3 3.x -> 4.x: bytes() is not serializable
            # also pandas sometimes returns int64
            # first-degree HexBytes and np.int64 check as a final trap
            for attr, value in mixed.items():
                if isinstance(value, HexBytes):
                    mixed[attr] = value.hex().lower()
                elif isinstance(value, np.int64):
                    mixed[attr] = int(value)

        return json.dumps(mixed)

    def _connect_redis(self, force_reconnect=False):
        """Connect to redis. Saves connection as self.redis."""
        if self.redis is None or force_reconnect is True:
            loc = urlparse(self.export_location)
            if loc.scheme == 'unix':
                unix_socket_path = loc.path
                conn = redis.Redis(unix_socket_path=unix_socket_path)
            else:
                hostname = loc.hostname
                port = loc.port
                if port is None:
                    port = 6379  # default redis port
                if loc.password is None:
                    conn = redis.Redis(host=hostname, port=int(port))
                else:
                    conn = redis.Redis(host=hostname, port=int(port),
                                       password=loc.password)
            self.redis = conn
        return self.redis

    def _get_export_options_from_settings(self):
        """Get export options from default settings."""
        if settings_file_loaded() is False:
            raise JSONExporterException("JSONExporter can't get settings.")
        export_type = get_setting("json", "output_type")
        self._check_export_type(export_type)
        self.export_type = export_type
        export_location = get_setting("json", "output_location")
        if export_type == 'file':
            export_location = os.path.abspath(export_location)
            if not os.path.isdir(export_location):
                # XXX should be on stderr
                print("WARN: Could not find output directory %s" % export_location)
                print("WARN: Making directory tree.")
                # attempt to make dirs
                os.makedirs(export_location, exist_ok=True)

        self.export_location = export_location


class JSONExporterException(Exception):
    """Generic exception for JSON Exporter."""
    pass
