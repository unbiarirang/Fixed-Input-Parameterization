import typer
import json


def main(input_path: str, output_path: str):
    """
    input_path: jsonl format
    output_path: jsonl format
    """
    input_data = []
    with open(input_path) as input_file:
        for line in input_file:
            input_data.append(json.loads(line))

    output_data = []
    for datum in input_data:
        for i, qa in enumerate(datum["qna"]):
            qa_id = datum["id"] + "-" + f"{i:05}"
            output_data.append(
                {
                    "id": qa_id,
                    "context": datum["context"],
                    "input": qa["question"],
                    "target": qa["answer"],
                }
            )

    with open(output_path, "w") as output_file:
        for datum in output_data:
            output_line = json.dumps(datum)
            print(output_line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
