import sys

from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler

from graph import build_graph


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else input("Topic: ")

    flow = build_graph()
    langfuse_handler = CallbackHandler()

    result = flow.invoke(
        {
            "topic": topic,
            "research": "",
            "summary": "",
            "critique": "",
            "approved": False,
            "iteration": 0,
        },
        config={"callbacks": [langfuse_handler]},
    )

    print(f"\n--- Summary (after {result['iteration']} iteration(s)) ---")
    print(result["summary"])


if __name__ == "__main__":
    main()
