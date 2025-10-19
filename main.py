from datasets import load_dataset, Dataset
from pydantic_ai import Agent, RunContext
from transformers import AutoTokenizer
import asyncio
from itertools import chain
from collections import Counter
import json
from math_verify import parse, verify


class MathDataset:
    def __init__(
        self,
        data_path: str = "AI-MO/aimo-validation-aime",
        model_name: str = "openai:gpt-5-mini",
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        ds = load_dataset(data_path, split="train")
        self.questions = [str(question) for question in ds["problem"]]
        self.answers = [str(answer) for answer in ds["answer"]]
        self.agent = Agent(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def num_unique_tokens(self) -> int:
        input_ids = self.tokenizer(self.questions)["input_ids"]
        return len(set(chain(*input_ids)))

    @property
    def token_counts(self) -> dict[str, int]:
        input_ids = self.tokenizer(self.questions)["input_ids"]
        return {self.tokenizer.decode(i): c for i, c in Counter(chain(*input_ids)).items()}


TRANSFORM_PROMPT = """\
Transform the following math question into a compact, line-based DSL that preserves the exact mathematics.

STRICT FIDELITY REQUIREMENTS
- Do NOT change numbers, variables, relations, equations, or assumptions.
- Preserve all names (points, variables, functions) exactly as given.
- Keep LaTeX/math semantics intact; only re-express wording.
- Output ONLY the DSL lines. No commentary, no markdown fences, no bullets.

DSL STYLE
- One fact/constraint per line.
- Prefer short tokens/symbols to reduce vocabulary.
- Allowed tokens (aim to stay within): shape, point, line, circle, angle, len, area, perimeter, radius, diameter, tangent, secant, intersects, parallel, equals, <, >, <=, >=, congruent, similar, midpoint, slope, product, sum, difference, ratio, gcd, lcm, prime, composite, integer, real, find.
- Use concise math forms: e.g., P(x)=2x^2+b*x+c, P(16)=54, area(ABCD)=m*sqrt(n).
- End with a single goal line starting with: "find ..."

FEW-SHOT EXAMPLES
before:
Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$
after:
P(x) = 2x^2 + b*x + c
Q(x) = -2x^2 + b*x + c
P(16) = 54
Q(16) = 54
P(20) = 53
Q(20) = 53
find P(0) + Q(0)

before:
Let $ABCD$ be a parallelogram with $\\angle BAD < 90^\\circ.$ A circle tangent to sides $\\overline{DA},$ $\\overline{AB},$ and $\\overline{BC}$ intersects diagonal $\\overline{AC}$ at points $P$ and $Q$ with $AP < AQ,$ as shown. Suppose that $AP=3,$ $PQ=9,$ and $QC=16.$ Then the area of $ABCD$ can be expressed in the form $m\\sqrt{n},$ where $m$ and $n$ are positive integers, and $n$ is not divisible by the square of any prime. Find $m+n.$
after:
shape ABCD
len(AB) = len(CD)
len(BC) = len(DA)
angle BAD < 90 degrees
circle E
E is tangent to DA
E is tangent to AB
E is tangent to BC
E intersects AC at P
E intersects AC at Q
AP < AQ
AP = 3
PQ = 9
QC = 16
area(ABCD) = m * sqrt(n)
m > 0
n > 0
n is squarefree
find m + n

NOW CONVERT THIS QUESTION
"""

SOLVE_PROMPT = "Solve the following question and provide your answer in \\boxed{}\n\n"""


async def map_agent(agent: Agent, prompts: list[str]) -> list[str]:
    futures = [agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*futures)
    return [result.output for result in results]


def grade(answers: list[str], preds: list[str]) -> float:
    grades = [verify(parse(answer), parse(pred)) for answer, pred in zip(answers, preds)]
    return sum(grades) / len(grades)


async def main() -> None:
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    questions = [str(question) for question in ds["problem"]]
    answers = [str(answer) for answer in ds["answer"]]

    agent = Agent("openai:gpt-5")
    
    transformed_questions = await map_agent(agent, [TRANSFORM_PROMPT + question for question in questions])

    base_llm_answers, transformed_llm_answers = await asyncio.gather( 
        map_agent(agent, [SOLVE_PROMPT + question for question in questions]),
        map_agent(agent, [SOLVE_PROMPT + question for question in transformed_questions]),
    )

    print(f"Base grade: {grade(answers, base_llm_answers):.2f}")
    print(f"Transformed grade: {grade(answers, transformed_llm_answers):.2f}")

    with open("data/v1.json", "w") as f:
        obj = [
            {
                "question": question,
                "answer": answer,
                "transformed_question": transformed_question,
                "base_llm_answer": base_llm_answer,
                "transformed_llm_answer": transformed_llm_answer,
            }
            for question, answer, transformed_question, base_llm_answer, transformed_llm_answer in zip(
                questions, 
                answers, 
                transformed_questions, 
                base_llm_answers, 
                transformed_llm_answers,
            )
        ]
        json.dump(obj, f)

    

    


