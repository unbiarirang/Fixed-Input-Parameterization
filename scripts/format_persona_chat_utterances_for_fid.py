import typer
import json

NUM_PERSONAS_PER_PREFIX = 5

def main(input_path: str, output_path: str, max_history: int = 0, tag: bool = False, long_prefix: bool = False):
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
        personas = datum["personality"]
        your_personas = [" ".join(personas[i*NUM_PERSONAS_PER_PREFIX: (i+1)*NUM_PERSONAS_PER_PREFIX]) \
                            for i in range(0, int(len(personas) / NUM_PERSONAS_PER_PREFIX))] \
                            if long_prefix else [" ".join(personas)]

        dialog_output_data = []
        for i, utterance in enumerate(datum["utterances"]):
            history = utterance["history"]
            candidates = utterance["candidates"]

            if not tag:
                history_tagged = [s for s in history[-(2*max_history+1):]]
            else:
                history_tagged = [("<parter>" if (len(history)-i) % 2 else "<you>") + ' ' + s for i, s in enumerate(history[-(2*max_history+1):])]
            index = f"{datum['id']}-{i:02}"
            ctxs = [{"title": "", "text": your_persona} for your_persona in your_personas]
            dialog_output_data.append({"id": index, "ctxs": ctxs, "question": " ".join(history_tagged), "target": candidates[-1]})

        output_data.extend(dialog_output_data)

    with open(output_path, "w") as output_file:
        for datum in output_data:
            line = json.dumps(datum)
            print(line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
