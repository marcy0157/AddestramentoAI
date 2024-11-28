import csv

def create_dataset(relevant_file, non_relevant_file, output_file):
    dataset = []

    # Funzione per processare i blocchi e aggiungere l'etichetta
    def process_file(file_path, relevance_label):
        current_block = {}
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line.startswith("Parola chiave:"):
                    current_block['Keyword'] = line.split("Parola chiave:")[1].strip()
                elif line.startswith("Titolo:"):
                    current_block['Title'] = line.split("Titolo:")[1].strip()
                elif line.startswith("Link:"):
                    current_block['Link'] = line.split("Link:")[1].strip()
                elif line.startswith("Descrizione:"):
                    current_block['Description'] = line.split("Descrizione:")[1].strip()
                elif line == "-----":
                    if current_block:
                        current_block['Relevance'] = relevance_label
                        dataset.append(current_block)
                        current_block = {}

    # Processa i due file
    process_file(relevant_file, "Relevant")
    process_file(non_relevant_file, "Not Relevant")

    # Salva il dataset in un file CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['Keyword', 'Title', 'Description', 'Link', 'Relevance']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Dataset creato con successo in: {output_file}")

# Percorsi dei file di input e output
relevant_file = 'relevant.txt'
non_relevant_file = 'non_relevant.txt'
output_file = 'dataset.csv'

# Creazione del dataset
create_dataset(relevant_file, non_relevant_file, output_file)
