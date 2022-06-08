import pandas as pd 
import pprint as pprint
import random
import typer

def main(persona_path: str, input_path: str, output_path: str):
    """
    input_path: personachat data file (txt)
    """
    TOTAL_NUM = [10,25,50] # Set this to vary the number of multiple personas in each input instance
    # getting unique personas
    df = pd.read_csv(persona_path)
    personas = []
    for index,row in df.iterrows():
        personas.append(row['persona'])

    original_df = pd.read_json(input_path, lines=True)
    def augment_personas(total_num):
        # augmenting to multiple personas
        entries = []
        for index,row in original_df.iterrows():
            id = row['id']
            personality = row['personality']
            utterances = row['utterances']
            while len(personality) < total_num:
                randn = random.randint(0,493)
                if personas[randn] not in personality:
                    personality.append(personas[randn])
            random.shuffle(personality)
            entry = {
                "id": id,
                "personality": personality,
                "utterances": utterances
            }
            entries.append(entry)
        pd.DataFrame(entries).to_json(f'{output_path}/benchmark_persona_chat_{str(total_num)}personas.jsonl', orient='records', lines=True)

    for t in TOTAL_NUM:
        augment_personas(t)

if __name__ == "__main__":
    typer.run(main)