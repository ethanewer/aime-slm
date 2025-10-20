import asyncio
import json
from collections import Counter
from itertools import chain

from math_verify import parse, verify
from pydantic_ai import Agent
from transformers import AutoTokenizer

TRANSFORM_PROMPT = """\
Reformat the math question so that they do not use any bad tokens. Use the `check_question` to find all bad tokens present in the question.

STRICT FIDELITY REQUIREMENTS
- Do NOT change numbers, variables, relations, equations, or assumptions.
- Keep math semantics intact.
- Output ONLY the DSL lines. No commentary, no markdown fences, no bullets.

DSL STYLE
- One fact/constraint per line.
- Prefer short tokens/symbols to reduce vocabulary.
- Allowed tokens (aim to stay within): shape, point, line, circle, angle, len, area, perimeter, radius, diameter, tangent, secant, intersects, parallel, equals, <, >, <=, >=, congruent, similar, midpoint, slope, product, sum, difference, ratio, gcd, lcm, prime, composite, integer, real, find.
- Use concise math forms: e.g., P(x)=2x^2+b*x+c, P(16)=54, area(ABCD)=m*sqrt(n).
- End with a single goal line starting with: "find ..."

NOW CONVERT THIS QUESTION
"""

SOLVE_PROMPT = "Solve the following question and provide your answer in \\boxed{}\n\n"


async def map_agent(agent: Agent, prompts: list[str]) -> list[str]:
    futures = [agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*futures)
    return [result.output for result in results]


def grade(answers: list[str], preds: list[str]) -> float:
    grades = [verify(parse(answer), parse(pred)) for answer, pred in zip(answers, preds)]
    return sum(grades) / len(grades)


async def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    with open("data/v1.json") as f:
        data = json.load(f)

    questions = [item["transformed_question"] for item in data]
    answers = [item["answer"] for item in data]
    input_ids = tokenizer(questions)["input_ids"]
    id_counts = Counter(chain(*input_ids))
    print(f"Before: {len(id_counts)} unique tokens.")

    agent = Agent("openai:gpt-5")

    @agent.tool
    def check_question(_, question: str) -> str:
        """Reports if a question satisfies the token requirements."""
        ids = tokenizer.encode(question)
        bad_ids_in_question = set(i for i in ids if id_counts[i] <= 1)
        if bad_ids_in_question:
            tokenized_question = [tokenizer.decode(i) for i in ids]
            bad_tokens_in_question = sorted(tokenizer.decode(i) for i in bad_ids_in_question)
            return (
                f"The question contains the following bad tokens: {bad_tokens_in_question}.\n"
                f"Here is the question split into tokens: {tokenized_question}\n"
                "You must continue to revise the question."
            )
        else:
            return "The question contains not bad tokens."

    print("Transforming questions.")
    transformed_questions = await map_agent(agent, [TRANSFORM_PROMPT + question for question in questions])

    print(f"After: {len(set(chain(*(tokenizer(transformed_questions)['input_ids']))))} unique tokens.")

    print("Solving questions.")
    transformed_llm_answers = await map_agent(agent, [SOLVE_PROMPT + question for question in transformed_questions])

    print(f"Transformed grade: {grade(answers, transformed_llm_answers):.2f}")

    with open("data/v2.json", "w") as f:
        obj = [
            {
                "question": question,
                "answer": answer,
                "transformed_question": transformed_question,
                "transformed_llm_answer": transformed_llm_answer,
            }
            for question, answer, transformed_question, transformed_llm_answer in zip(
                questions,
                answers,
                transformed_questions,
                transformed_llm_answers,
            )
        ]
        json.dump(obj, f)


if __name__ == "__main__":
    asyncio.run(main())
