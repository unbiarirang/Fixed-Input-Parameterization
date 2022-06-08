import typer
import json

def main(input_path: str, split: str, output_path: str):
    """
    input_path: jsonl
    split: train or valid
    """
    
    with open(input_path) as input_file:
        dataset = json.load(input_file)

    input_data = dataset[split]

    output_data = []
    for i, datum in enumerate(input_data):
        output_data.append({"id": f"{i:0>5}", **datum})
            
    with open(output_path, "w") as output_file:
        for datum in output_data:
            line = json.dumps(datum)
            print(line, file=output_file)


if __name__ == "__main__":
    typer.run(main)