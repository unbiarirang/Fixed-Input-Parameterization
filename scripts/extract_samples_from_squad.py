from datasets import load_dataset
from collections import Counter
import random
import typer
import json


def main(
    output_path: str,
    num_min_samples: int = 20,
    sample: bool = False,
    num_samples: int = 20,
    seed: int = 42,
):

    d = load_dataset("squad")
    valid = d["validation"]

    indices = []
    titles = []
    contexts = []
    source_texts = []
    target_texts = []
    for data in valid:
        title, context, question, answer, index = (
            data["title"],
            data["context"],
            data["question"],
            data["answers"]["text"][0],
            data["id"],
        )
        if context not in contexts:
            indices.append(index)
            titles.append(title)
            contexts.append(context)
            source_texts.append([question])
            target_texts.append([answer])
        else:
            idx = contexts.index(context)
            source_texts[idx] += [question]
            target_texts[idx] += [answer]
    assert len(contexts) == len(titles)
    assert len(contexts) == len(source_texts)
    assert len(contexts) == len(target_texts)

    remain_flags = [True if len(x) >= num_min_samples else False for x in source_texts]
    indices = [x for x, flag in zip(indices, remain_flags) if flag]
    titles = [x for x, flag in zip(titles, remain_flags) if flag]
    contexts = [x for x, flag in zip(contexts, remain_flags) if flag]
    question_lists = [x for x, flag in zip(source_texts, remain_flags) if flag]
    answer_lists = [x for x, flag in zip(target_texts, remain_flags) if flag]

    if sample:
        random.seed(seed)
        test_idxs = random.sample(range(len(question_lists)), num_samples)
        remain_flags = [True if idx in test_idxs else False for idx in range(len(question_lists))]

        indices = [x for x, flag in zip(indices, remain_flags) if flag]
        titles = [x for x, flag in zip(titles, remain_flags) if flag]
        contexts = [x for x, flag in zip(contexts, remain_flags) if flag]
        question_lists = [x for x, flag in zip(question_lists, remain_flags) if flag]
        answer_lists = [x for x, flag in zip(answer_lists, remain_flags) if flag]

    print("Number of contexts:", len(contexts))
    for i, q in zip(indices, question_lists):
        print("Context", i, "Number of questions", len(q))

    with open(output_path, "w") as output_file:
        for id, title, context, question_list, answer_list in zip(
            indices, titles, contexts, question_lists, answer_lists
        ):
            output_line = json.dumps(
                {
                    "id": id,
                    "title": title,
                    "context": context,
                    "qna": [
                        {"question": question, "answer": answer}
                        for question, answer in zip(question_list, answer_list)
                    ],
                }
            )
            print(output_line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
