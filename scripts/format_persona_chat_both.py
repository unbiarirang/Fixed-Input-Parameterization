import typer
import json

def main(input_path: str, output_path: str, max_history: int = 0, tag: bool = False, last: bool = False):
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
    persona_index = 0
    for datum in input_data:
        your_persona = " ".join(datum["your_persona"])
        parter_persona = " ".join(datum["parter_persona"])

        your_persona_index = persona_index
        persona_index += 1
        parter_persona_index = persona_index
        persona_index += 1

        # add empty last turn
        if last:
            if all([(turn["parter"] != "" and turn["you"] != "") for turn in datum["dialog"]]):
                datum["dialog"].append({"parter": "", "you": ""})

        history = []
        dialog_output_data = []
        for turn in datum["dialog"]:
            parter_utterance, your_utterance = turn["parter"], turn["you"]
            if not tag:
                history_tagged = [s for s in history[-(2*max_history+1):]]
            else:
                history_tagged = [("<parter>" if (len(history)-i) % 2 else "<you>") + ' ' + s for i, s in enumerate(history[-(2*max_history+1):])]
            index = f"{parter_persona_index:05}-{len(history):02}"
            dialog_output_data.append({"id": index, "context": parter_persona, "input": " ".join(history_tagged), "target": parter_utterance})
            history.append(parter_utterance)

            if parter_utterance != "":
                if not tag:
                    history_tagged = [s for s in history[-(2*max_history+1):]]
                else:
                    history_tagged = [("<parter>" if (len(history)-i) % 2 else "<you>") + ' ' + s for i, s in enumerate(history[-(2*max_history+1):])]
                index = f"{your_persona_index:05}-{len(history):02}"
                dialog_output_data.append({"id": index, "context": your_persona, "input": " ".join(history_tagged), "target": your_utterance})
                history.append(your_utterance)

        if not last:
            output_data.extend(dialog_output_data)
        else:
            output_data.append(dialog_output_data[-1])
            
    with open(output_path, "w") as output_file:
        for datum in output_data:
            line = json.dumps(datum)
            print(line, file=output_file)


if __name__ == "__main__":
    typer.run(main)