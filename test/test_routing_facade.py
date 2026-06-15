from uni_api.routing.index import build_api_key_models_map, build_routing_index
from uni_api.routing.planner import RoutingPlanner
from uni_api.routing.scheduler import lottery_scheduling, weighted_round_robin


def test_routing_facade_exposes_index_and_scheduler_helpers():
    config = {
        "providers": [{"provider": "p1", "model": ["gpt-4.1"]}],
        "api_keys": [{"api": "sk-test", "model": ["p1/gpt-4.1"]}],
    }

    index = build_routing_index(config, ["sk-test"])

    assert index.models_by_provider["p1"] == ("gpt-4.1",)
    assert build_api_key_models_map(config, ["sk-test"]) == {"sk-test": ["gpt-4.1"]}
    assert weighted_round_robin({"a": 2, "b": 1}).count("a") == 2
    assert len(lottery_scheduling({"a": 2, "b": 1})) == 3


def test_routing_planner_facade_exists():
    assert hasattr(RoutingPlanner(), "plan")
