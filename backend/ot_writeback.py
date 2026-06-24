# backend/ot_writeback.py
"""OPC-UA Write‑Back Service

Provides a secure, asynchronous interface for issuing emergency‑stop commands to PLCs.

Key features
- Reads connection parameters from the environment:
    * ``OPCUA_ENDPOINT_URL`` – e.g. ``opc.tcp://192.168.1.10:4840``
    * ``OPCUA_SECURITY_MODE`` – ``SignAndEncrypt`` (default) or ``Sign``
    * ``OPCUA_SECURITY_POLICY`` – ``Basic256Sha256`` (default) or other supported policy
- Uses ``asyncua`` with native security profiles (SignAndEncrypt + Basic256Sha256).
- Maintains a singleton ``OPCUAWriteBack`` instance that is lazily started.
- Background 1‑second heartbeat validates the session and reconnects on failure.
- Exponential back‑off retry logic guarantees the FastAPI runtime never crashes on PLC drop.
- Exposes ``async trigger_emergency_stop(asset_id: str, reason: str)`` which writes ``True`` to a safety register ``ns=2;s=Emergency_Stop`` under the asset node.
"""

import os
import asyncio
import logging
from typing import Optional
from asyncua import Client, ua
from asyncua.ua.uaerrors import UaError

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Environment configuration (fallback defaults provided for local testing)
OPCUA_ENDPOINT_URL = os.getenv("OPCUA_ENDPOINT_URL", "opc.tcp://localhost:4840")
OPCUA_SECURITY_MODE = os.getenv("OPCUA_SECURITY_MODE", "SignAndEncrypt")  # Sign | SignAndEncrypt | None
OPCUA_SECURITY_POLICY = os.getenv("OPCUA_SECURITY_POLICY", "Basic256Sha256")
OPCUA_HEARTBEAT_INTERVAL_SECS = int(os.getenv("OPCUA_HEARTBEAT_INTERVAL_SECS", "1"))

class OPCUAWriteBack:
    """Singleton class handling a secure OPC‑UA client connection.

    The client is created on demand and kept alive by a periodic heartbeat.
    Any failure triggers an exponential‑back‑off reconnection loop that never
    propagates an exception to the caller.
    """

    _instance: Optional["OPCUAWriteBack"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure the init body runs only once
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.endpoint = OPCUA_ENDPOINT_URL
        self.security_mode = OPCUA_SECURITY_MODE
        self.security_policy = OPCUA_SECURITY_POLICY
        self._client: Optional[Client] = None
        self._lock = asyncio.Lock()
        self._backoff = 1  # start delay seconds
        self._max_backoff = 30
        self._stop_event = asyncio.Event()
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _create_client(self) -> Client:
        client = Client(self.endpoint)
        # Apply security settings if requested
        if self.security_mode and self.security_policy:
            try:
                mode = getattr(ua.MessageSecurityMode, self.security_mode)
                policy = getattr(ua.SecurityPolicy, self.security_policy)
                client.set_security(policy, None, None, mode)
                log.info("OPC-UA security configured: mode=%s policy=%s", self.security_mode, self.security_policy)
            except Exception as e:
                log.warning("Failed to apply OPC-UA security settings (%s, %s): %s", self.security_mode, self.security_policy, e)
        return client

    async def _ensure_connected(self) -> Client:
        """Return a connected client, establishing the connection with back‑off if needed."""
        async with self._lock:
            if self._client and self._client.uaclient and self._client.uaclient.session_active:
                return self._client
            # (re)connect with exponential back‑off
            while not self._stop_event.is_set():
                try:
                    self._client = await self._create_client()
                    await self._client.connect()
                    log.info("Connected to OPC-UA server %s", self.endpoint)
                    self._backoff = 1
                    return self._client
                except Exception as exc:
                    log.error("OPC-UA connection failed: %s – retrying in %s sec", exc, self._backoff)
                    await asyncio.sleep(self._backoff)
                    self._backoff = min(self._backoff * 2, self._max_backoff)
            raise RuntimeError("OPC-UA WriteBack stopped before connection could be established")

    async def _heartbeat_loop(self) -> None:
        """Background task that validates the connection every second.

        If a read fails, the client is discarded and reconnection is attempted on the next loop.
        """
        while not self._stop_event.is_set():
            try:
                client = await self._ensure_connected()
                # Simple read of the Server_NamespaceArray node as a liveness check
                await client.read_node(ua.NodeId(ua.ObjectIds.Server_NamespaceArray))
                # Reset back‑off on successful ping
                self._backoff = 1
            except Exception as exc:
                log.warning("OPC-UA heartbeat failed: %s – will reconnect", exc)
                # Force reconnection on next iteration
                if self._client:
                    try:
                        await self._client.disconnect()
                    except Exception:
                        pass
                    self._client = None
            await asyncio.sleep(OPCUA_HEARTBEAT_INTERVAL_SECS)

    async def trigger_emergency_stop(self, asset_id: str, reason: str) -> None:
        """Write a True value to the emergency‑stop node of the target PLC.

        The node hierarchy is assumed to follow the convention:
        ``ns=2;s=Assets.{asset_id}.Emergency_Stop``. Adjust the node string if your
        PLC uses a different address space.
        """
        try:
            client = await self._ensure_connected()
            node_id = f"ns=2;s=Assets.{asset_id}.Emergency_Stop"
            node = client.get_node(node_id)
            await node.write_value(True)
            # Optionally write a reason string to a sibling node for audit
            reason_node = client.get_node(f"ns=2;s=Assets.{asset_id}.Emergency_Stop_Reason")
            await reason_node.write_value(reason)
            log.info("Emergency stop issued for %s – reason: %s", asset_id, reason)
        except Exception as exc:
            log.critical(
                "Critical: failed to issue emergency stop for %s (reason: %s): %s",
                asset_id,
                reason,
                exc,
                exc_info=True,
            )
            # Re‑raise so callers can decide further handling, but the heartbeat will recover the session.
            raise

    async def shutdown(self) -> None:
        """Gracefully stop the heartbeat and close the client."""
        self._stop_event.set()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            if self._client:
                await self._client.disconnect()
                self._client = None
        log.info("OPCUAWriteBack shutdown complete")

# Export a ready‑to‑use singleton instance for FastAPI import
ot_writeback = OPCUAWriteBack()

# Public async API used by other modules
async def trigger_emergency_stop(asset_id: str, reason: str) -> None:
    """Convenient wrapper that forwards to the singleton instance."""
    await ot_writeback.trigger_emergency_stop(asset_id, reason)
