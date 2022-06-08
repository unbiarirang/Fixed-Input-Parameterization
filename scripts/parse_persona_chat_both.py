import typer
import json
from collections import defaultdict


def main(input_path: str, output_path: str):
    """
    input_path: personachat data file (txt)
    """
    with open(input_path) as input_file:
        persona_chat = input_file.readlines()

    parsed_data = []
    your_persona = []
    parter_persona = []
    dialog = []
    prev_n = 0
    for line in persona_chat:
        n, text = line.split(" ", maxsplit=1)
        if n == "1" and len(dialog) > 0:
            parsed_data.append(
                {"your_persona": your_persona, "parter_persona": parter_persona, "dialog": dialog}
            )
            your_persona = []
            parter_persona = []
            dialog = []
            prev_n = 0
        assert int(n) == prev_n + 1
        prev_n += 1

        if text.startswith("your persona: "):
            content = text[len("your persona: ") :].strip()
            your_persona.append(content)
            continue
        if text.startswith("partner's persona: "):
            content = text[len("partner's persona: ") :].strip()
            parter_persona.append(content)
            continue

        two_turns, _ = text.split("\t\t")  # turn, candidates
        parter_utterance, your_utterance = two_turns.split("\t")
        dialog.append({"parter": parter_utterance, "you": your_utterance})
    else:
        parsed_data.append(
            {"your_persona": your_persona, "parter_persona": parter_persona, "dialog": dialog}
        )

    with open(output_path, "w") as output_file:
        for datum in parsed_data:
            line = json.dumps(datum)
            print(line, file=output_file)


if __name__ == "__main__":
    typer.run(main)
