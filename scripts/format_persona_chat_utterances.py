import typer
import json
from random import shuffle

def main(input_path: str, output_path: str, max_history: int = 0, permute: bool = False, repeat: int = 1, tag: bool = False):
    """
    input_path: jsonl
    last: only take last turn where there is no reply. Not used anymore.
    """
    assert tag == True or max_history == 0

    input_data = []
    with open(input_path) as input_file:
        for line in input_file:
            input_data.append(json.loads(line))

    output_data = []
    for datum in input_data:
        persona = datum["personality"].copy()

        for _ in range(repeat):
            for i, utterance in enumerate(datum["utterances"]):
                if permute:
                    shuffle(persona)
                your_persona = " ".join(persona)
                
                history = utterance["history"]
                candidates = utterance["candidates"]

                if not tag:
                    history_tagged = [s for s in history[-(2*max_history+1):]]
                else:
                    history_tagged = [("<parter>" if (len(history)-i) % 2 else "<you>") + ' ' + s for i, s in enumerate(history[-(2*max_history+1):])]
                index = f"{datum['id']}-{i:02}"
                output_data.append({"id": index, "context": your_persona, "input": " ".join(history_tagged), "target": candidates[-1]})
            
    with open(output_path, "w") as output_file:
        for datum in output_data:
            line = json.dumps(datum)
            print(line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
