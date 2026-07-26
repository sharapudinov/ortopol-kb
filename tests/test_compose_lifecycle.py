"""Unit tests for deploy/compose_lifecycle.py: no Docker.
Split out of test_deploy_units.py (module size).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import compose_lifecycle


class WaitHealthyTests(unittest.TestCase):
    """Injects clock/sleep/is_healthy so the timing-dependent poll loop is
    exercised without a real Docker daemon or real sleeping.
    """

    def _fake_clock(self, start=0.0):
        state = {"t": start}

        def clock():
            return state["t"]

        def sleep(seconds):
            state["t"] += seconds

        return clock, sleep

    def test_healthy_on_first_poll(self):
        clock, sleep = self._fake_clock()
        ok = compose_lifecycle.wait_healthy(
            "proj", {}, "svc", timeout=30, clock=clock, sleep=sleep, is_healthy=lambda: True,
        )
        self.assertTrue(ok)

    def test_healthy_on_second_poll(self):
        clock, sleep = self._fake_clock()
        calls = {"n": 0}

        def is_healthy():
            calls["n"] += 1
            return calls["n"] >= 2

        ok = compose_lifecycle.wait_healthy(
            "proj", {}, "svc", timeout=30, clock=clock, sleep=sleep, is_healthy=is_healthy,
        )
        self.assertTrue(ok)
        self.assertEqual(calls["n"], 2)

    def test_never_healthy_times_out(self):
        clock, sleep = self._fake_clock()
        ok = compose_lifecycle.wait_healthy(
            "proj", {}, "svc", timeout=10, clock=clock, sleep=sleep, is_healthy=lambda: False,
        )
        self.assertFalse(ok)


class DefaultIsHealthyClosureTests(unittest.TestCase):
    """9cad8fe9: is_healthy=None -- the path every real caller uses (see
    smoke_test.py, which never passes is_healthy itself) -- builds its own
    closure over container_id()/health_status() that WaitHealthyTests above
    never exercises, since every one of those tests supplies its own
    is_healthy. container_id/health_status are mocked at the module level
    (the closure calls them as compose_lifecycle.container_id/
    health_status, not through a local reference) so the composition logic
    itself -- not real Docker -- is what's under test.
    """

    def _fake_clock(self, start=0.0):
        state = {"t": start}

        def clock():
            return state["t"]

        def sleep(seconds):
            state["t"] += seconds

        return clock, sleep

    def test_no_container_is_unhealthy_without_calling_health_status(self):
        # bool(cid) short-circuits before health_status(cid) is ever called
        # -- calling docker inspect on a cid of None would be nonsensical.
        clock, sleep = self._fake_clock()
        with mock.patch.object(compose_lifecycle, "container_id", return_value=None), \
             mock.patch.object(compose_lifecycle, "health_status") as health_status_mock:
            ok = compose_lifecycle.wait_healthy("proj", {}, "svc", timeout=10, clock=clock, sleep=sleep)
        self.assertFalse(ok)
        health_status_mock.assert_not_called()

    def test_container_present_but_not_healthy_is_unhealthy(self):
        clock, sleep = self._fake_clock()
        with mock.patch.object(compose_lifecycle, "container_id", return_value="abc123"), \
             mock.patch.object(compose_lifecycle, "health_status", return_value="starting"):
            ok = compose_lifecycle.wait_healthy("proj", {}, "svc", timeout=10, clock=clock, sleep=sleep)
        self.assertFalse(ok)

    def test_container_present_and_healthy_is_healthy(self):
        clock, sleep = self._fake_clock()
        with mock.patch.object(compose_lifecycle, "container_id", return_value="abc123") as cid_mock, \
             mock.patch.object(compose_lifecycle, "health_status", return_value="healthy") as health_mock:
            ok = compose_lifecycle.wait_healthy("proj", {}, "svc", timeout=10, clock=clock, sleep=sleep)
        self.assertTrue(ok)
        cid_mock.assert_called_with("proj", {}, "svc", compose_file=compose_lifecycle.COMPOSE_FILE)
        health_mock.assert_called_with("abc123")


if __name__ == "__main__":
    unittest.main()
