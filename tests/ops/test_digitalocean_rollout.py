"""Offline safety checks; never contact Docker, SSH or production endpoints."""

import importlib.util
from pathlib import Path
import shlex
import types
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/digitalocean-rollout.py"
spec = importlib.util.spec_from_file_location("rollout", SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class RolloutSafety(unittest.TestCase):
    def setUp(self):
        self.remote = {"__name__": "rollout_test"}
        exec(compile(wrapper.REMOTE_SCRIPT, "remote_rollout", "exec"), self.remote)
        self.error = self.remote["RolloutError"]

    def test_never_stop_the_serving_container(self):
        for port, name in [(8001, "uni-api"), (8002, "uni-api-management-candidate")]:
            run = Mock()
            with patch.dict(self.remote, current_origin_port=lambda: port, run=run):
                with self.assertRaises(self.error):
                    self.remote["stop_clean"](name)
            run.assert_not_called()

    def test_drain_has_no_forced_kill_timeout(self):
        run = Mock()
        with patch.dict(self.remote, current_origin_port=lambda: 8002, run=run,
                        inspect=lambda _: {"State": {"ExitCode": 0, "OOMKilled": False}}):
            self.remote["stop_clean"]("uni-api")
        run.assert_called_once_with(["docker", "stop", "-t", "-1", "uni-api"])

    def test_unclean_exit_is_not_accepted(self):
        with patch.dict(self.remote, current_origin_port=lambda: 8002, run=Mock(),
                        inspect=lambda _: {"State": {"ExitCode": 137, "OOMKilled": False}}):
            with self.assertRaises(self.error):
                self.remote["stop_clean"]("uni-api")

    def test_check_only_checks_serving_commit(self):
        with patch.dict(self.remote,
                        inspect=lambda _: {"Config": {"Env": ["SOURCE_COMMIT=old"]}},
                        image_info=lambda _: {}, current_origin_port=lambda: 8001,
                        control_summary=Mock(), health=lambda _: True):
            with self.assertRaisesRegex(self.error, "expected release"):
                self.remote["check_only"]("latest", "new")

    def simulate(self, *, serving=8001, active_candidate=False, expected="new",
                 retained_error=False, ready_error=False, compose_error=False):
        events = []
        state = {"port": serving, "primary": "old", "running": True,
                 "candidate": active_candidate}
        image = {"id": "new-image", "source_commit": "new"}

        def inspect(name, **kwargs):
            if name == "uni-api":
                return {"Image": state["primary"], "State": {"Running": state["running"]}}
            if state["candidate"]:
                return {"Image": "new-image", "State": {"Running": True}}
            return None

        def run(command):
            events.append(command)
            if command[:2] == ["docker", "compose"]:
                if compose_error:
                    raise self.error("compose failed")
                state.update(primary="new-image", running=True)

        def stop(name):
            self.assertNotEqual(state["port"], 8001 if name == "uni-api" else 8002)
            events.append(["stop", name])
            state["running" if name == "uni-api" else "candidate"] = False

        def switch(policy, old, new):
            self.assertEqual(state["port"], old)
            events.append(["switch", old, new])
            state["port"] = new

        def ready(name, *args, **kwargs):
            if ready_error:
                raise self.error("restore not ready")
            return {"intent": {"rules": ["preserved"]}}

        def retained(port):
            if retained_error:
                raise self.error("saved intent differs")

        args = types.SimpleNamespace(image="latest", expected_commit=expected, ready_timeout=1)
        replacements = dict(
            inspect=inspect, compose_metadata=lambda _: {}, COMPOSE=SCRIPT, CONFIG=SCRIPT,
            image_info=lambda _: image, current_origin_port=lambda: state["port"], run=run,
            choose_policy=lambda: ("policy", None), retained_equal=retained,
            write_candidate_env=Mock(), start_candidate=lambda *a: state.update(candidate=True),
            wait_ready=ready, control_summary=ready, health=lambda _: True,
            switch=switch, stop_clean=stop, final_report=Mock(),
        )
        error = None
        with patch.dict(self.remote, **replacements):
            try:
                self.remote["rollout"](args)
            except self.error as exc:
                error = str(exc)
        return state, events, error

    def test_failed_preconditions_never_stop_primary(self):
        for options in [{"expected": "wrong"}, {"retained_error": True}, {"ready_error": True}]:
            state, events, error = self.simulate(**options)
            self.assertIsNotNone(error)
            self.assertEqual(state["port"], 8001)
            self.assertTrue(state["running"])
            self.assertFalse(any(event[0] in ("switch", "stop") for event in events))

    def test_failed_replacement_preserves_serving_candidate(self):
        state, events, error = self.simulate(compose_error=True)
        self.assertIsNotNone(error)
        self.assertEqual(state["port"], 8002)
        self.assertTrue(state["candidate"])
        self.assertNotIn(["stop", "uni-api-management-candidate"], events)

    def test_success_switches_before_each_drain(self):
        state, events, error = self.simulate()
        self.assertIsNone(error)
        self.assertEqual(state["port"], 8001)
        self.assertEqual(state["primary"], "new-image")
        self.assertFalse(state["candidate"])
        for port, name in [(8002, "uni-api"), (8001, "uni-api-management-candidate")]:
            self.assertLess(events.index(["switch", 8001 if port == 8002 else 8002, port]),
                            events.index(["stop", name]))

    def test_resume_does_not_pull_over_active_candidate(self):
        state, events, error = self.simulate(serving=8002, active_candidate=True)
        self.assertIsNone(error)
        self.assertFalse(any(event[:2] == ["docker", "pull"] for event in events))
        self.assertEqual(state["port"], 8001)

    def test_ssh_arguments_are_quoted_and_options_forwarded(self):
        run = Mock(return_value=types.SimpleNamespace(returncode=0))
        with patch("sys.argv", [str(SCRIPT), "--image", "image; echo unsafe", "--check-only",
                                "--ssh-option=KexAlgorithms=curve25519-sha256"]), \
                patch.object(wrapper.subprocess, "run", run):
            self.assertEqual(wrapper.main(), 0)
        command = run.call_args.args[0]
        self.assertIn("KexAlgorithms=curve25519-sha256", command)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[remote.index("--image") + 1], "image; echo unsafe")
        self.assertEqual(remote[remote.index("--expected-commit") + 1], "")
        self.assertIn("--check-only", remote)


if __name__ == "__main__":
    unittest.main()
