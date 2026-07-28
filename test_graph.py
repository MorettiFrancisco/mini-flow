from langgraph.graph import END

from graph import MAX_ITERATIONS, route_after_critique


def test_route_after_critique():
    assert route_after_critique({"approved": True, "iteration": 1}) == END
    assert route_after_critique({"approved": False, "iteration": 1}) == "summarize"
    assert route_after_critique({"approved": False, "iteration": MAX_ITERATIONS}) == END


if __name__ == "__main__":
    test_route_after_critique()
    print("ok")
