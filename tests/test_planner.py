from augmentor.agents.planner import make_plan


def test_make_plan_balances():
    plan = make_plan({0: 10, 1: 5, 2: 0}, target_coverage=100, max_new_per_class=100)
    assert plan.quotas[1] == 5
    assert plan.quotas[2] == 10
