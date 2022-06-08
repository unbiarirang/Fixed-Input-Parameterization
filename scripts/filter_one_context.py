import typer
import json


def main(input_path: str, context_id: str, output_path: str):
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
        if datum["id"].startswith(context_id):
            output_data.append(datum)

    with open(output_path, "w") as output_file:
        for datum in output_data:
            output_line = json.dumps(datum)
            print(output_line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
